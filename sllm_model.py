# pip install hf_transfer accelerate peft
# pip install langchain==0.3.27 langchain-core==0.3.76 langchain-community==0.3.30 langchain-text-splitters==0.3.11 langchain-huggingface==0.3.1 langchain-ollama==0.3.10
# pip install torch torchvision torchaudio transformers sentence-transformers faiss-cpu

import os, time, torch, platform, re, json 
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path

from peft import PeftModel
from langchain.vectorstores import FAISS
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import MessagesPlaceholder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").strip()
PROMPTS = {
    "summarizer": (PROMPT_DIR / "summarizer.txt").read_text(encoding="utf-8").strip(),
    "task_extractor": (PROMPT_DIR / "extract_tasks.txt").read_text(encoding="utf-8").strip(),
}

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
ft_model_name = "CHOROROK/Qwen2.5_1.5B_trained_model_v3"

# ===== 이스케이트 ====
def escape_curly(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

# ===== 벡터 DB 로드 =====
def load_faiss_db(db_path: str):
    embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("🔵 FAISS DB 로드 완료!\n")
    return vector_store, embedding_model


# ===== 모델 로드 =====
def load_model_q(model_name, adapter_name: str | None = None):
    if platform.system() == "Windows":
        print("⚠ Windows에서는 4bit 불가 → FP16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        print("🔵 Linux/RunPod 환경: 4bit 없이 bf16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,   # 안 되면 torch.float16 로 바꿔도 됨
            device_map="auto",
        )

    if adapter_name:
        print(f"🔵 LoRA/PEFT 어댑터 로드: {adapter_name}")
        model = PeftModel.from_pretrained(base_model, adapter_name)
    else:
        model = base_model

    text_gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, 
        max_new_tokens = 2048, 
        temperature=0.2,
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipe)
    return llm


# ===== 도메인 필터 =====
def make_filter(filter: dict):
    if any(filter.values()):
        main_filter = filter.copy()
    else:
        main_filter = None
    return main_filter


# ===== RAG 단어 추출 – 현재 안씀 =====
def make_rag_result(model, meeting_text):
    instruction = """
    당신은 회의록 전문을 분석하는 AI입니다. 의미가 모호한 단어를 모두 중복없이 추출하세요.
    - 의미가 모호한 용어는 절대 추측하지 않고 그대로 추출
    - 일반 인사, 잡담은 제외
    - 출력은 콤마로 구분된 단어 목록으로 해주세요.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction),
        HumanMessagePromptTemplate.from_template("회의록: {text}")
    ])

    formatted_prompt = prompt.format(text=meeting_text)
    output = model(formatted_prompt, temperature=0.2, top_p=0.9)

    rag_word_list = [w.strip() for w in re.split(r'[, \n]+', output) if w.strip()]
    print("🔹 모르는 단어 리스트:", rag_word_list)
    return rag_word_list


# ===== 용어 추출용 체인 =====
def build_term_extractor_chain(llm: ChatHuggingFace):
    """회의록에서 모호한/핵심 용어를 콤마로 추출하는 체인."""
    instruction = """
    당신은 IT 회의록을 분석하는 전문가입니다.
    아래 회의록에서 '정의가 필요해 보이는 용어'를 5~15개 정도 뽑아주세요.

    기준:
    - 서비스/기능 이름, 기술 용어, 약어, 지표/지수, 정책/규칙 이름 등
    - 일반적인 일상어(안녕하세요, 네, 좋아요 등)는 제외
    - 이미 너무 명확한 단어(예: 로그인, 버튼)도 웬만하면 제외
    - 출력은 오직 콤마로 구분된 용어 리스트만 반환하세요. 예)
      회원가입 SSO, 작업 보드 CRUD, RICE 스코어, CI/CD 품질 게이트
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(instruction),
            HumanMessagePromptTemplate.from_template("회의록 전문:\n{text}"),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


# ===== definitions =====
class DefinitionAgent:

    def __init__(self, llm: ChatHuggingFace, vector_store: FAISS, default_domain: str = "IT"):
        self.llm = llm
        self.default_domain = default_domain

        # 벡터스토어 retriever 준비
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.65,
                "filter": {"domain": default_domain},
            },
        )

        # 용어 추출 체인
        self.term_chain = build_term_extractor_chain(llm)

    def invoke(self, inputs: dict):
        """LangChain AgentExecutor와 맞추기 위해 .invoke(dict)를 제공."""
        text = inputs.get("input", "")
        if not text:
            return {"output": json.dumps({"definitions": {}}, ensure_ascii=False)}

        # 추출
        terms_text = self.term_chain.invoke({"text": str(text or "")})
        term_list = [t.strip() for t in re.split(r"[,;\n]+", terms_text) if t.strip()]

        print("🔹🔹🔹type of text:", type(text), text)
        print("🔹 에이전트 추출 용어:", term_list)

        # term 정의 검색
        definitions = {}
        for term in term_list:
            docs = self.retriever.invoke(term)
            if not docs:
                # 못 찾은 용어는 패ㅡ
                continue

            # 가장 관련도 높은 문서 1~2개를 합쳐서 정의로 사용
            defs = []
            for d in docs[:2]:
                ans = d.metadata.get("answer") or d.page_content
                defs.append(ans.strip())

            definitions[term] = "\n\n".join(defs)

        # 3) JSON 문자열로 반환
        return {
            "output": json.dumps({"definitions": definitions}, ensure_ascii=False)
        }


def build_agent(llm, vector_store, default_domain="IT"):
    return DefinitionAgent(llm, vector_store, default_domain)


def make_chain(model):
    summarizer_prompt = PROMPTS["summarizer"]
    task_prompt = PROMPTS["task_extractor"]  
    
    safe_summarizer = escape_curly(PROMPTS["summarizer"])
    safe_task_prompt = escape_curly(PROMPTS["task_extractor"])
 
    instruction = ("""
    [SYSTEM_PROMPT]
        [안건 / 요약 지침]
        다음 내용을 회의록 안건 추출 및 요약 파트에 적용하라:
        {{safe_summarizer}}

        [태스크 추출 지침]
        다음 내용을 tasks 추출 파트에 적용하라:
        {{safe_task_prompt}}

    [공통 출력 규칙]
    - 최종 출력은 반드시 하나의 JSON 문자열만 반환한다.
    - 불필요한 자연어 설명, 앞뒤 인사말, 코드 블록 마크다운(````json` 등)은 절대 넣지 않는다.
    - keys를 중복 정의하지 않는다. (예: "tasks"를 두 번 쓰지 말 것)
    - definitions(용어 정의)는 참고만 하고, summary/tasks/issues에 그대로 장문 복붙하지 말 것.

    출력 스키마(예시):

    {{
      "agendas": [
        {{
        "agenda_1": {{
        "who": "...",
        "what": "...",
        "when": "...",
        "where": "...",
        "why": "...",
        "how": "...",
        "how_much": "...",
        "how_long": "..."
        }},
        "agenda_2": {{
          "who": "...",
          "what": "...",
          "when": "...",
          "where": "...",
          "why": "...",
          "how": "...",
          "how_much": "...",
          "how_long": "..."
        }}],
      "tasks": [
        {{
          "owner": "이름 또는 역할",
          "task": "해야 할 일",
          "due": "YYYY-MM-DD 또는 'TBD' 혹은 빈 문자열"
        }}
      ]
    }}
    """)

    # instruction = _escape_curly(instruction)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction),
        HumanMessagePromptTemplate.from_template(
            "사용자 회의록: {text}\n\n"
            "참고 문서:\n{rag_result_text}\n\n"
            "위 System 프롬프트와 각 역할별 프롬프트 지침을 모두 반영하여,\n"
            "반드시 하나의 JSON만 생성하세요."
        )
    ])
    chain = prompt | model | parser
    return chain

def run_inference_model(transcript: str):
    load_dotenv()
    db_path = './faiss_db/rag_it_tta'
    HF_TOKEN = os.getenv('HF_TOKEN')

    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device set to:", device)

    vector_store, embedding_model = load_faiss_db(db_path)
    base_model = load_model_q(base_model_name)
    ft_model = load_model_q(base_model_name, adapter_name = ft_model_name)

    agent = build_agent(base_model, vector_store, default_domain='IT')

    chain = make_chain(ft_model)
    print("회의록 전문을 입력하세요! 종료하려면 'exit' 입력\n")

    while True:
        query = input("전문: ")
        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        print("\n--- 🔍 에이전트: 단어 정의 추출중 ---")
        agent_result = agent.invoke({"input": query})
        rag_result_text = agent_result["output"]
        print(" 🔍 에이전트 definitions:", rag_result_text)

        print("\n--- 🤖 파튜 모델: 요약 생성중 ---")
        result = chain.invoke({
            "text": query,
            "rag_result_text": rag_result_text
        })

        # AgentExecutor는 보통 {"output": "...", ...} 형태 반환
        print("\n모델 응답(JSON):\n", result)

    return {"success": True, "data": {"summary": result['agedas'], "tasks": result['tasks'],}}


if __name__ == "__main__":
    q = input('전문: ')
    result_final = run_inference_model(q)
    print("\n모델 응답(JSON):\n", result_final)