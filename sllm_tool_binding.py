import json
import torch
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from main_model import preprocess_transcript

# ===== Configuration =====
CACHE_DIR = "/workspace/hf_cache"
# BASE_MODEL_NAME = "Qwen/Qwen3-8B"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-instruct"
# FT_MODEL_NAME = ""
FT_MODEL_NAME = "CHOROROK/Qwen2.5_1.5B_meeting_agenda_task"
DB_PATH = "./faiss_db_merged"

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").strip()
PROMPTS = {
    "summarizer": (PROMPT_DIR / "summarizer.txt").read_text(encoding="utf-8").strip(),
    "task_extractor": (PROMPT_DIR / "extract_tasks.txt").read_text(encoding="utf-8").strip(),
}


# 채팅 보내는 메시지 구조
@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


# 정의할 툴 내부 구조
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class QwenLLMAgent:
    def __init__(
        self,
        model_id: str = BASE_MODEL_NAME,
        adapter_id: Optional[str] = FT_MODEL_NAME,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        system_prompt: str = SYSTEM_PROMPT
    ):

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False, 
            cache_dir=CACHE_DIR
        )

        # 기본 모델 로드
        print(f"📦 모델 로드: {model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch_dtype == "auto" else torch_dtype,
            device_map=device_map,
            cache_dir=CACHE_DIR
        )
        # 어댑터 로드
        if adapter_id:
            print(f"🔧 어댑터 로드: {adapter_id}")
            self.model = PeftModel.from_pretrained(base_model, adapter_id)
        else:
            self.model = base_model

        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {}

        print("✅ 모델 로드 완료!\n")

    # 커스텀 프롬프트 설정
    def set_system_prompt(self, system_prompt):
        """시스템 프롬프트를 업데이트하고 기존 시스템 메시지를 제거합니다."""
        self.system_prompt = system_prompt
        user_messages = [m for m in self.messages if m.role != 'system']
        self.messages = user_messages

    # 에이전트가 사용할 툴 등록
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        """툴을 등록합니다."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        self.tools[name] = tool
        print(f"✅ Registered tool: {name}")

    # 프롬프트에 삽입할 툴 설명들
    def _build_tool_descriptions(self) -> str:
        """사용 가능한 툴의 설명을 생성합니다."""
        if not self.tools:
            return ""

        tool_list = []
        for tool in self.tools.values():
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            tool_list.append(tool_info)

        tools_text = "\n\n[Available Tools]\n"
        tools_text += json.dumps(tool_list, indent=2, ensure_ascii=False)
        tools_text += "\n\n[Tool Usage Rules]\n"
        tools_text += "1. ONLY use the 'retrieval' tool when you find domain-specific or technical terms in the PROVIDED TRANSCRIPT that are ambiguous.\n"
        tools_text += "2. NEVER search for terms that are NOT in the transcript.\n"
        tools_text += "3. NEVER search for common words or dates (like 'due_date', 'current_date', etc.).\n"
        tools_text += "4. Extract terms directly from the transcript text ONLY.\n"
        tools_text += "5. If no ambiguous domain terms exist in the transcript, do NOT use the tool.\n\n"
        tools_text += "[Tool Usage Format]\n"
        tools_text += 'To use a tool, respond with JSON in this format:\n'
        tools_text += '{"tool": "tool_name", "parameters": {...}}\n'
        tools_text += "After receiving tool results, provide your final answer.\n"

        return tools_text

    def _clean_text(self, text: str) -> str:
        """서로게이트 문자 및 문제가 되는 유니코드 문자를 제거합니다."""
        if not isinstance(text, str):
            text = str(text)

        # 서로게이트 문자 제거
        try:
            # UTF-8로 인코딩 가능한지 확인하고, 불가능한 문자 제거
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            # 폴백: 안전한 문자만 유지
            text = ''.join(char for char in text if ord(char) < 0x10000)

        return text

    def _format_messages(self) -> List[Dict[str, str]]:
        """메시지를 Qwen 모델 형식으로 변환합니다."""
        formatted = []

        # 툴 설명과 함께 시스템 프롬프트 추가
        system_content = self._clean_text(self.system_prompt)

        if self.tools:
            system_content += "\n" + self._clean_text(self._build_tool_descriptions())

        formatted.append({
            "role": "system",
            "content": system_content
        })

        # 모든 메시지 히스토리 추가 (전문을 볼 수 있도록)
        for msg in self.messages:
            formatted.append({
                "role": msg.role,
                "content": self._clean_text(msg.content)
            })

        return formatted

    # 호출된 툴 실행시키기
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """툴을 실행하고 결과를 반환합니다."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        try:
            tool = self.tools[tool_name]
            result = tool.function(**parameters)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    # 모델 응답에서 툴 호출 파싱
    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 JSON 툴 호출을 파싱합니다."""
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                return None

            json_str = text[start_idx:end_idx + 1]
            tool_call = json.loads(json_str)

            if "tool" in tool_call and "parameters" in tool_call:
                return tool_call

            return None
        except json.JSONDecodeError:
            return None

    # 채팅
    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_tools: bool = True
    ) -> str:
        """사용자 입력을 받아 응답을 생성합니다."""
        
        if user_input is not None and not isinstance(user_input, str):
            user_input = json.dumps(user_input, ensure_ascii=False)
            print(type(user_input))

        # 빈 입력이 아닐 때만 메시지 추가 (툴 호출 후 재귀에서는 빈 문자열)
        if user_input:
            self.messages.append(Message(role="user", content=user_input))

        formatted_messages = self._format_messages()

        # 템플릿 삽입 및 토큰화
        model_inputs = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # apply_chat_template가 tensor만 주는 버전 대비
        if isinstance(model_inputs, torch.Tensor):
            model_inputs = {"input_ids": model_inputs}

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        print(type(model_inputs))


        # 응답 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )

        # 생성된 응답 추출
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()

        # 응답 디코딩
        content = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        ).strip("\n")

        # 툴 호출 (use_tools=True일 때만)
        tool_call = self._parse_tool_call(content) if use_tools else None

        if tool_call and self.tools and use_tools:
            # 툴 실행시키기
            tool_name = tool_call["tool"]
            parameters = tool_call["parameters"]

            print(f"\n[Tool Call: {tool_name}]")
            print(f"Parameters: {parameters}")

            tool_result = self._execute_tool(tool_name, parameters)
            print(f"Result: {tool_result}\n")

            # 툴 호출 결과 히스토리에 저장
            self.messages.append(Message(
                role="assistant",
                content=content,
                tool_calls=[tool_call]
            ))

            self.messages.append(Message(
                role="user",
                content=f"Tool '{tool_name}' returned: {tool_result}\n\nBased on this result, please provide your final answer."
            ))

            # 응답 재생성하기
            return self.chat(
                "",  # 빈 문자열 - 이미 메시지 추가됨
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_tools=use_tools
            )

        # 히스토리에 응답 저장
        self.messages.append(Message(role="assistant", content=content))

        return content

    def clear_history(self):
        """채팅 히스토리를 초기화합니다."""
        self.messages = []
        print("Chat history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """채팅 히스토리를 반환합니다."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]


# ===== Retrieval Tool 생성 함수 =====
def create_retrieval_tool(db_path: str = DB_PATH, domain_filter: Optional[str] = None):
    """FAISS DB를 로드하고 검색 함수를 반환합니다."""

    # 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        encode_kwargs={'normalize_embeddings': True}
    )

    # FAISS DB 로드
    vector_store = FAISS.load_local(
        db_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    if domain_filter:
        print(f"🔵 FAISS DB 로드 완료: {db_path} (도메인 필터: {domain_filter})\n")
    else:
        print(f"🔵 FAISS DB 로드 완료: {db_path}\n")

    def retrieval_search(term_list: List[str], k: int = 3, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """
        단어 리스트를 받아 각 단어의 정의를 검색합니다.

        Args:
            term_list: 검색할 도메인 특화 용어 리스트
            k: 각 용어당 검색할 결과 개수
            domain: 검색할 도메인 필터 (지정하지 않으면 domain_filter 사용)

        Returns:
            {term: [definition1, definition2, ...]} 형태의 딕셔너리
        """
        results = {}

        # 도메인 필터 결정 (파라미터 > 생성 시 필터 > None)
        filter_domain = domain or domain_filter

        for term in term_list:
            # 도메인 필터가 있으면 메타데이터 필터링 적용
            if filter_domain:
                # FAISS 검색 수행 (메타데이터 필터링)
                docs = vector_store.similarity_search(
                    term,
                    k=k,
                    filter={"domain": filter_domain}
                )
                # k개만 선택
                docs = docs[:k]
            else:
                # 필터 없이 검색
                docs = []

            definitions = [doc.page_content for doc in docs]
            results[term] = definitions

            # 검색 결과 로그
            if filter_domain:
                print(f"  🔍 '{term}' 검색 (도메인: {filter_domain}) - {len(definitions)}개 결과")
            else:
                print(f"  🔍 도메인 미검색")

        return results

    return retrieval_search


# ===== 회의록 처리 파이프라인 =====
def process_meeting_transcript(
    transcript: str,
    agent: QwenLLMAgent,
    current_date: str = None,
    use_retrieval: bool = True,
    domain: Optional[str] = None
) -> Dict[str, Any]:

    # 현재 날짜 설정
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        print("오늘 날짜 !!!!!!: ", current_date)

    # Step 0: 전문 이해 (Retrieval Tool 사용)
    if use_retrieval and agent.tools:
        print("=" * 50)
        print("STEP 0: 전문 이해 중 (도메인 용어 검색)...")
        print("=" * 50)

        understanding_prompt = f"""다음 회의록 전문을 읽고, 의미가 모호한 도메인 특화 용어나 기술 용어가 있다면 retrieval tool을 사용해 검색하세요.

[회의록 전문]
{transcript}

전문에 도메인 특화 용어가 있으면 검색하고, 없으면 "이해 완료"라고 답하세요."""

        agent.clear_history()
        understanding_response = agent.chat(understanding_prompt, temperature=0.3, use_tools=True)
        print(f"\n전문 이해 결과: {understanding_response}\n")

    # Step 1: 안건 추출
    print("=" * 50)
    print("STEP 1: 안건 추출 중...")
    print("=" * 50)

    summarizer_prompt = PROMPTS["summarizer"].format(transcript=transcript)
    agent.clear_history()
    agendas_response = agent.chat(summarizer_prompt, temperature=0.2, use_tools=False)

    try:
        agendas = json.loads(agendas_response)
    except json.JSONDecodeError:
        print("⚠️ 안건 추출 JSON 파싱 실패")
        agendas = {"agendas": []}

    print(f"\n✅ 안건 추출 완료: {len(agendas.get('agendas', []))}개\n")

    # Step 2: 태스크 추출
    print("=" * 50)
    print("STEP 2: 태스크 추출 중...")
    print("=" * 50)

    # 현재 날짜로부터 요일 계산
    date_obj = datetime.strptime(current_date, "%Y-%m-%d")
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    current_weekday = weekdays[date_obj.weekday()]

    task_prompt = PROMPTS["task_extractor"].format(
        current_date=current_date,
        current_weekday=current_weekday,
        transcript=transcript
    )
    agent.clear_history()
    tasks_response = agent.chat(task_prompt, temperature=0.2, use_tools=False)

    try:
        tasks = json.loads(tasks_response)
    except json.JSONDecodeError:
        print("⚠️ 태스크 추출 JSON 파싱 실패")
        tasks = {"tasks": []}

    print(f"\n✅ 태스크 추출 완료: {len(tasks.get('tasks', []))}개\n")

    # 결과 합치기
    result = {
        "agendas": agendas.get("agendas", []),
        "tasks": tasks.get("tasks", [])
    }

    return result


def agent_main(domain_input, transcript):

    # Agent 초기화
    print("=" * 50)
    print("QwenLLMAgent 초기화 중...")
    print("=" * 50)

    agent = QwenLLMAgent(
        model_id=BASE_MODEL_NAME,
        adapter_id=FT_MODEL_NAME
    )

    # 도메인 입력
    domain_filter = domain_input if domain_input else None

    # Retrieval Tool 등록
    print("\n" + "=" * 50)
    print("Retrieval Tool 등록 중...")
    print("=" * 50)

    retrieval_func = create_retrieval_tool(DB_PATH, domain_filter=domain_filter)
    agent.register_tool(
        name="retrieval",
        description="회의록 전문에서 의미가 모호한 도메인 특화 단어를 검색하는 도구입니다.",
        parameters={
            "type": "object",
            "properties": {
                "term_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "검색할 도메인 용어 리스트"
                },
                "k": {
                    "type": "integer",
                    "description": "각 용어당 검색할 결과 개수",
                    "default": 1
                },
                "domain": {
                    "type": "string",
                    "description": "검색할 도메인 (예: IT, 의료, 법률)",
                    "default": domain_filter
                }
            },
            "required": ["term_list"]
        },
        function=retrieval_func
    )

    print("\n" + "=" * 50)
    print("회의록 처리 파이프라인 시작")
    print("=" * 50)
    result = process_meeting_transcript(
        # transcript=sample_transcript,
        transcript=transcript,
        agent=agent,
        current_date=None,
        use_retrieval=True,  # 전문 이해 시 retrieval tool 사용
        domain=domain_filter
    )
    return result



if __name__ == "__main__":
    domain_input = input("\n검색할 도메인을 입력하세요 (예: IT, 의료, 법률, 엔터없이 입력 시 전체 검색): ").strip()

    # 회의록 샘플
    sample_transcript = input("\n회의록 전문을 입력하세요:\n")
    sample_transcript = preprocess_transcript(sample_transcript)
    result = agent_main(domain_input, sample_transcript)
    print("\n" + "=" * 50)
    print("최종 결과:")
    print("=" * 50)
    print(json.dumps(result, ensure_ascii=False, indent=2))