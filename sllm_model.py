import os
import json
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate

from main_model import load_model_q, load_faiss_db, escape_curly

# ===== 기본설정 =====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

db_path = './faiss_db_merged'
vector_store, embedding_model = load_faiss_db(db_path)

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").strip()
PROMPTS = {
    "summarizer": (PROMPT_DIR / "summarizer.txt").read_text(encoding="utf-8").strip(),
    "task_extractor": (PROMPT_DIR / "extract_tasks.txt").read_text(encoding="utf-8").strip(),
}

summarizer_prompt = PROMPTS["summarizer"]
task_prompt = PROMPTS["task_extractor"]


# ===== 청크 분할 (1500자 기준 앞 뒤 문장) =====
def chunk_transcript(transcript: str, max_tokens: int = 1500) -> List[str]:
    # 문장 단위로 분할 (한국어 문장 종결 기준)
    sentences = []
    current = ""
    for char in transcript:
        current += char
        if char in ['.', '!', '?', '\n'] or (char == '다' and len(current) > 20):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    # 청크 생성
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += " " + sentence
            current_length += sentence_length

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ===== 청크 별 처리 및 전체 요약/태스크 추출 =====
def process_transcript_with_chunks(agent, transcript: str, max_chunk_tokens: int = 1500) -> dict:
    
    print(f"\n{'='*60}")
    print(f"전문 길이: {len(transcript)} 글자")
    print(f"{'='*60}\n")

    # 전문 길 경우 - 청크 
    if len(transcript) > max_chunk_tokens:
        chunks = chunk_transcript(transcript, max_chunk_tokens)
        print(f"전문을 {len(chunks)}개 청크로 분할했습니다.\n")

        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[청크 {i}/{len(chunks)}] 처리 중... (길이: {len(chunk)} 글자)")
            try:
                # 각 청크 agent 처리 (용어 검색)
                result = agent.invoke({"input": f"다음은 회의록의 일부입니다. 이해하기 어려운 전문 용어가 있으면 검색해주세요:\n\n{chunk}"})
                chunk_results.append({"chunk_index": i, "chunk_length": len(chunk), "result": result.get("output", "")})
                print(f"[청크 {i}] 처리 완료")
            except Exception as e:
                print(f"[청크 {i}] 처리 실패: {e}")
                chunk_results.append({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "error": str(e)
                })
    else:
      # 전문 길이 적절 시 - 풀로 진행
      print("전문 길이가 적절합니다. 청크 분할 없이 진행합니다.\n")
      chunk_results = []

    # 전체 전문 기반으로 세부 안건, 안건별 요약 추출
    print(f"\n{'='*60}")
    print("전체 전문 기반 안건/요약 추출 중...")
    print(f"{'='*60}\n")

    try:
        summary_result = agent.invoke({"input": summarizer_prompt})
        full_summary = summary_result.get("output", "")
        print("✅ 안건/요약 추출 완료\n")
    except Exception as e:
        print(f"❌ 안건/요약 추출 실패: {e}\n")
        full_summary = {"error": str(e)}

    print(f"\n{'='*60}")
    print("전체 전문 기반 태스크 추출 중...")
    print(f"{'='*60}\n")

    try:
        task_result = agent.invoke({"input": task_prompt})
        full_tasks = task_result.get("output", "")
        print("✅ 태스크 추출 완료\n")
    except Exception as e:
        print(f"❌ 태스크 추출 실패: {e}\n")
        full_tasks = {"error": str(e)}

    return {"chunk_results": chunk_results, "full_summary": full_summary, "full_tasks": full_tasks}


# ===== agent!!! =====
def build_agent(model, vector_store, domain) :

    safe_summarizer = escape_curly(PROMPTS["summarizer"])
    safe_task_prompt = escape_curly(PROMPTS["task_extractor"])

    search_kwargs = {"k": 20}
    if domain:
        search_kwargs["filter"] = {"domain": domain}

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    template = '''
        You are an AI meeting-analysis agent specialized in {domain} projects.
        You will receive a meeting transcript about {domain} topics.
        (e.g., Analysis of the latest market trends and competitor movements, Project Kick-off Meeting).

        You must answer as accurately as possible using the available tools.
        You have access to the following tools:
        {tools}

        Your primary goals when handling a meeting-related request are:
        1) Understand the meeting context:
           - 목적(purpose), 참여자(participants), 결정사항(decisions), 미해결 이슈(open issues)를 파악한다.
        2) When necessary, clarify or look up IT/technical/domain terms or concepts using tools.
        3) From the meeting transcript, you must be able to:
           - extract detailed agenda list
           - summarize the meeting
           - extract follow-up tasks
        4) Ground your answers in the meeting transcript and retrieved domain documents; 
           NEVER hallucinate requirements or decisions that are not supported by the content.

          - Every field MUST be filled ONLY with words/phrases that appear in the original transcript.
          - If a certain 5W3H item is not explicitly stated in the transcript, set that field to null.
          - Do NOT invent or infer new facts that are not present in the transcript.

        Use the following ReAct-style format:

        transcript: the input transcript for the meeting
        Thought: you should always think about what to do next
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat up to 15 times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question in Korean
        (When you need JSON outputs, internally follow the Summary JSON Prompt
         or Tasks JSON Prompt described above.)

        Hard constraints (format rules – MUST NOT be violated):
        - Immediately after any line that starts with "Thought:", the very next line MUST be one of the following:
          1) "Action: ..."
          2) "Final Answer: ..."
        - Do NOT write bullet points, long explanations, or any additional sentences between "Thought:" and
          the next "Action:" or "Final Answer:". The line immediately following "Thought:" must be exactly
          one of those two formats.
        - When you use a tool, you MUST follow this format exactly:
          Thought: ...
          Action: tool_name
          Action Input: "the input to pass to the tool"
          Observation: the result returned by the tool
        - When you no longer need to use any tools and you want to finish the answer, you MUST follow this format:
          Thought: I can now provide the final answer.
          Final Answer: (write the final answer in Korean)

        Important rules:
        - If the user request is general chit-chat, a simple greeting, or a very simple question, you MAY skip Action/Action Input/Observation and respond directly with Final Answer.
        - If you need additional domain knowledge or definitions, choose the most appropriate tool from [{tool_names}] and use it.
        - Use the meeting transcript and retrieved documents as the primary source of truth.
        - When you summarize or extract issues/decisions/tasks, be faithful to the transcript.
        - Final Answer MUST be written in Korean, unless the user clearly asks for another language.

        Begin!

        transcript:{input}
        Thought:{agent_scratchpad}
        domain:{domain}
        '''

    prompt = PromptTemplate.from_template(template).partial(domain=domain)

    # retrieval tool
    def retrieval_func(term_list: str) -> dict:
        """
        FAISS 벡터스토어에서 전문 중 모르는 단어를 검색해 단어 정의를 반환하는 툴.
        Args: term_list: 검색할 용어들 (단일 문자열 또는 JSON 형식의 리스트 문자열)
             예: "API" 또는 '["API", "마이크로서비스"]'
        Returns: 검색된 용어들의 정의를 담은 딕셔너리
        """
        print('='*60)
        print('🔍 retrieval_func 호출됨')
        print('입력값:', term_list)
        print('입력 타입:', type(term_list))
        print("현재 domain 필터 = ", domain)
        print('='*60)

        # 입력 정규화
        try:
            if isinstance(term_list, str):
                term_list = term_list.strip()
                if term_list.startswith('[') and term_list.endswith(']'):
                    try:
                        term_list_local = json.loads(term_list)
                        if not isinstance(term_list_local, list):
                            term_list_local = [str(term_list_local)]
                    except json.JSONDecodeError as je:
                        print(f'⚠️ JSON 파싱 실패: {je}, 문자열 그대로 사용')
                        # 대괄호 제거하고 쉼표로 split
                        term_list_local = [t.strip().strip('"\'') for t in term_list.strip('[]').split(',')]
                elif ',' in term_list:
                    term_list_local = [t.strip().strip('"\'') for t in term_list.split(',')]
                else:
                    term_list_local = [term_list]

            elif isinstance(term_list, list):
                term_list_local = term_list

            elif isinstance(term_list, tuple):
                term_list_local = list(term_list)

            else:
                print(f'⚠️ 예상치 못한 타입: {type(term_list)}, 문자열로 변환')
                term_list_local = [str(term_list)]

            print('파싱된 term_list_local:', term_list_local)
            term_list_local = [str(term).strip().strip('"\'') for term in term_list_local if term]
            term_list_local = [t for t in term_list_local if t]
            print('최종 정리된 term_list_local:', term_list_local)

        except Exception as parse_error:
            print(f'❌ 입력 파싱 중 오류: {type(parse_error).__name__}: {parse_error}')
            print(f'원본 입력을 단일 항목으로 처리합니다: {repr(term_list)}')
            term_list_local = [str(term_list).strip()]

        if not term_list_local:
            print('검색할 용어가 없습니다.')
            return {"output": json.dumps({"definitions": {}}, ensure_ascii=False)}

        # 개별 검색 처리
        all_docs_list = []
        k = search_kwargs.get("k", 20)
        filter_dict = search_kwargs.get("filter", None)

        for idx, term in enumerate(term_list_local):
            try:
                # term을 명시적으로 문자열로 변환
                term_str = str(term).strip()
                print(f'\n[{idx+1}/{len(term_list_local)}] 검색 시작')
                print(f'  원본 term: {repr(term)} (타입: {type(term).__name__})')
                print(f'  변환 term_str: {repr(term_str)} (타입: {type(term_str).__name__})')

                if not term_str:
                    print(f'⚠️ 빈 검색어 건너뜀')
                    all_docs_list.append([])
                    continue

                print(f'  🔍 FAISS 검색 실행: "{term_str}"')
                print(f'  검색 파라미터: k={k}, filter={filter_dict}')

                # query 타입 재확인
                if not isinstance(term_str, str):
                    raise TypeError(f"query must be str, got {type(term_str).__name__}")

                if len(term_str) == 0:
                    print(f'⚠️ 빈 문자열, 건너뜀')
                    all_docs_list.append([])
                    continue

                # FAISS 검색 실행
                docs = vector_store.similarity_search(
                    term_str,  # 키워드 인자 대신 위치 인자 사용
                    k=k,
                    filter=filter_dict
                )
                all_docs_list.append(docs)
                print(f'  ✅ 검색 완료: {len(docs)}개 문서 발견')
            
            except Exception as e:
                print(f'  ❌ 검색 실패!')
                print(f'     오류 타입: {type(e).__name__}')
                print(f'     오류 메시지: {e}')
                print(f'     문제된 term: {repr(term)}')
                
                import traceback
                traceback.print_exc()
                all_docs_list.append([])

        definitions = {}
        for term, docs in zip(term_list_local, all_docs_list):
            if not docs:
                print(f'@#$@#$@${term}에 대한 용어 못찾음@#$#@$@#')
                continue

            defs = []
            for d in docs[:2]:
                ans = d.metadata.get("answer") or d.page_content
                defs.append(ans.strip())

            definitions[term] = "\n\n".join(defs)

        print('여기다 여기!!! 단어 정의 : ', definitions)
        retrieval_result = {"output": json.dumps({"definitions": definitions}, ensure_ascii=False)}

        return retrieval_result

    # Tool 객체 직접 생성
    from pydantic import BaseModel, Field

    class RetrievalInput(BaseModel):
        term_list: str = Field(description="검색할 용어들 (단일 문자열 또는 JSON 배열 문자열)")

    retrieval_tool = Tool(
        name="retrieval",
        func=retrieval_func,
        description="FAISS 벡터스토어에서 전문 중 모르는 단어를 검색해 단어 정의를 반환하는 툴. 입력은 JSON 배열 형식의 문자열 또는 단일 용어 문자열.",
        args_schema=RetrievalInput
    )

    tools = [retrieval_tool]

    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20, max_execution_time=400, handle_parsing_errors=True)

    return agent_executor


