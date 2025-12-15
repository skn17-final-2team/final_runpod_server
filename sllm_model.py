import os
import json
# import platform
from pathlib import Path
# from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import traceback

# from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from datetime import datetime

from main_model import load_model_q, load_faiss_db, preprocess_transcript

# ===== 기본설정 =====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

db_path = "./faiss_db_merged"
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
def process_transcript_with_chunks(transcript: str, domain) -> dict:

    user_domain = domain
    if not user_domain :
        domain_filter = None
    else:
        domain_filter = user_domain

    # 전문 문장으로 전처리
    transcript = preprocess_transcript(transcript)

    # 에이전트 빌드
    agent = build_agent(model=load_model_q(), vector_store=vector_store, domain=domain_filter)

    # 긴 회의록 처리를 위한 청크 크기 설정
    max_chunk_len = 3000

    # 전문 길이 확인 후 청크 분할 여부 결정
    if len(transcript) > max_chunk_len:
        chunks = chunk_transcript(transcript, max_chunk_len)
        print(f"⚠️ 전문이 깁니다! {len(chunks)}개 청크로 분할하여 처리합니다.\n")

        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[청크 {i}/{len(chunks)}] 처리 중... (길이: {len(chunk)} 글자)")
            try:
                # 각 청크별로 전문 용어 검색 (선택적)
                result = agent.invoke({"input": f"다음은 회의록의 일부입니다. 이해하기 어려운 전문 용어가 있으면 최대 5개까지만 검색해주세요:\n\n{chunk}"})
                chunk_results.append({"chunk_index": i, "chunk_length": len(chunk), "result": result.get("output", "")})
                print(f"[청크 {i}] 처리 완료")
            except Exception as e:
                print(f"[청크 {i}] 처리 실패: {e}")
                chunk_results.append({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "error": str(e)
                })

        # 청크가 너무 많으면 전체 전문을 요약본으로 축약
        if len(chunks) > 5:
            print(f"\n⚠️ 청크가 {len(chunks)}개로 너무 많습니다. 전체 요약/태스크 추출 시 청크별 요약을 사용합니다.")
            # 청크별 결과를 합쳐서 축약된 전문 생성
            condensed_transcript = "\n\n".join([
                f"[청크 {cr['chunk_index']}] {cr.get('result', '')[:500]}..."
                for cr in chunk_results if 'result' in cr
            ])
            use_full_transcript = False
        else:
            condensed_transcript = transcript
            use_full_transcript = True
    else:
        # 전문 길이 적절 시 - 풀로 진행
        print("✅ 전문 길이가 적절합니다. 청크 분할 없이 진행합니다.\n")
        chunk_results = []
        condensed_transcript = transcript
        use_full_transcript = True

    # 전체 전문 기반으로 세부 안건, 안건별 요약 추출
    print(f"\n{'='*60}")
    print("전체 전문 기반 안건/요약 추출 중...")
    print(f"{'='*60}\n")

# ===== 안건/요약 추출 =====
    try:
        transcript_for_analysis = condensed_transcript if not use_full_transcript else transcript
        filled_summary_prompt = f"""
            다음 회의 전문을 분석하세요.
            {summarizer_prompt.format(transcript=transcript_for_analysis)}
            
            CRITICAL INSTRUCTIONS:
            1. ONLY use information EXPLICITLY mentioned in the transcript above
            2. DO NOT invent or hallucinate any information
            3. If you need to look up unknown terms, use the retrieval tool with at most 5 SHORT terms
            4. Your final answer MUST start with "Final Answer:" followed by the JSON
            5. DO NOT include any explanations before or after the JSON"""

        summary_result = agent.invoke({"input": filled_summary_prompt})
        full_summary = summary_result.get("output", "")

        if "Agent stopped" in full_summary or "iteration limit" in full_summary:   # Agent가 iteration limit에 도달한 경우 intermediate_steps에서 결과 추출
            print("⚠️ Agent iteration limit 도달, 중간 결과 추출 시도...")
            intermediate_steps = summary_result.get("intermediate_steps", [])
            
            # # if intermediate_steps:
            # #     # 마지막 agent 출력에서 JSON 추출

            # # 모든 step 출력 확인 (디버깅)
            # print(f"   총 {len(intermediate_steps)}개의 intermediate steps 발견")

            # 모든 LLM 출력에서 JSON 찾기
            for i, step in enumerate(intermediate_steps):
                if len(step) >= 1:
                    # step은 (AgentAction, observation) 튜플
                    agent_action = step[0] if len(step) > 0 else None

                    # AgentAction의 log에서 JSON 추출
                    if hasattr(agent_action, 'log'):
                        log_text = str(agent_action.log)
                        if '{' in log_text and '"agendas"' in log_text:
                            # JSON 부분만 추출
                            start_idx = log_text.find('{')
                            end_idx = log_text.rfind('}') + 1
                            if start_idx != -1 and end_idx > start_idx:
                                json_candidate = log_text[start_idx:end_idx]
                                # 유효성 검증
                                try:
                                    import json as json_module
                                    parsed = json_module.loads(json_candidate)
                                    if 'agendas' in parsed:
                                        full_summary = json_candidate
                                        print(f"✅ Step {i}에서 유효한 JSON 추출 성공!")
                                        break
                                except:
                                    continue

            # 위 방법이 실패하면 마지막 출력에서 추출
            if "Agent stopped" in full_summary:
                for step in reversed(intermediate_steps):
                    if len(step) > 0 and hasattr(step[0], 'log'):
                        log_text = str(step[0].log)
                        if '{' in log_text and '}' in log_text:
                            full_summary = log_text.strip()
                            print(f"✅ 폴백: 마지막 step에서 텍스트 추출")
                # for step in reversed(intermediate_steps):
                    # if len(step) > 1 and hasattr(step[1], '__str__'):
                    #     potential_output = str(step[1])
                    #     if '{' in potential_output and '}' in potential_output:
                    #         full_summary = potential_output
                    #         print(f"✅ 중간 결과 추출 성공 (길이: {len(full_summary)}자)")
                    #         break

        print("✅ 안건/요약 추출 완료\n")
    except Exception as e:
        print(f"❌ 안건/요약 추출 실패: {e}\n")
        full_summary = {"error": str(e)}

    print(f"\n{'='*60}")
    print("전체 전문 기반 태스크 추출 중...")
    print(f"{'='*60}\n")

# ===== 태스크 추출 =====   
    try:
        transcript_for_analysis = condensed_transcript if not use_full_transcript else transcript
        current_date=datetime.now().date().isoformat()
        date_obj = datetime.strptime(current_date, "%Y-%m-%d")
        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        current_weekday=weekdays[date_obj.weekday()]
        
        filled_task_prompt = f"""
            다음 회의 전문을 분석하세요.
    
            {task_prompt.format(transcript=transcript_for_analysis, current_date=datetime.now().date().isoformat(), current_weekday=current_weekday)}
            
            CRITICAL INSTRUCTIONS:
            1. ONLY extract tasks and assignees that are EXPLICITLY mentioned in the transcript
            2. DO NOT invent names like "김영희" - ONLY use names that appear in the transcript
            3. If an assignee is not clearly stated, use null
            4. If you need to look up unknown terms, use the retrieval tool with at most 5 SHORT terms
            5. Your final answer MUST start with "Final Answer:" followed by the JSON
            6. DO NOT include any explanations before or after the JSON"""

        task_result = agent.invoke({"input": filled_task_prompt})
        full_tasks = task_result.get("output", "")

        # Agent가 iteration limit에 도달한 경우 intermediate_steps에서 결과 추출
        if "Agent stopped" in full_tasks or "iteration limit" in full_tasks:
            print("⚠️ Agent iteration limit 도달, 중간 결과 추출 시도...")
            intermediate_steps = task_result.get("intermediate_steps", [])
            if intermediate_steps:
                # 마지막 agent 출력에서 JSON 추출
                for step in reversed(intermediate_steps):
                    if len(step) > 1 and hasattr(step[1], '__str__'):
                        potential_output = str(step[1])
                        if '{' in potential_output and '}' in potential_output:
                            full_tasks = potential_output
                            print(f"✅ 중간 결과 추출 성공 (길이: {len(full_tasks)}자)")
                            break

        print("✅ 태스크 추출 완료\n")
    except Exception as e:
        print(f"❌ 태스크 추출 실패: {e}\n")
        full_tasks = {"error": str(e)}

    # return {"chunk_results": chunk_results, "full_summary": full_summary, "full_tasks": full_tasks}
    return {"full_summary": full_summary, "full_tasks": full_tasks}


# ===== agent!!! =====
def build_agent(model, vector_store, domain) :

    search_kwargs = {"k": 20}
    if domain:
        search_kwargs["filter"] = {"domain": domain}

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    template = '''
    AI meeting analyzer for {domain}.

    Tools: {tools}  

    FORMAT (MANDATORY):
    Thought: [reasoning]
    Action: [{tool_names}] or "None"
    Action Input: [input] or "N/A"
    Observation: [result]
    ...(repeat if needed)
    Thought: I now know the final answer
    Final Answer: [Korean JSON] 

    RULES:
    1. ALWAYS end with "Final Answer:" - NO exceptions
    2. Use retrieval ONLY for unknown terms (max 5, under 50 chars each): ["term1","term2"]
    3. Use ONLY explicit transcript info - NO hallucinations
    4. Keep it concise  

    Input: {input}
    Thought:{agent_scratchpad}'''

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

        # 최대 검색어 길이 제한 (임베딩 모델의 토큰 제한 고려)
        MAX_TERM_LENGTH = 100  # 한 용어당 최대 100자

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

            # 너무 긴 검색어 필터링 및 경고
            filtered_terms = []
            for t in term_list_local:
                if t:
                    if len(t) > MAX_TERM_LENGTH:
                        print(f'⚠️ 검색어가 너무 깁니다 (길이: {len(t)}). 앞 {MAX_TERM_LENGTH}자만 사용: "{t[:MAX_TERM_LENGTH]}..."')
                        filtered_terms.append(t[:MAX_TERM_LENGTH])
                    else:
                        filtered_terms.append(t)

            term_list_local = filtered_terms
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

                # 추가 안전장치: 검색어가 너무 긴 경우 다시 한번 확인
                if len(term_str) > MAX_TERM_LENGTH:
                    print(f'⚠️ 검색어 재확인: 길이 {len(term_str)} > {MAX_TERM_LENGTH}, 잘라냄')
                    term_str = term_str[:MAX_TERM_LENGTH]

                # FAISS 검색 실행
                try:
                    docs = vector_store.similarity_search(
                        term_str,  # 키워드 인자 대신 위치 인자 사용
                        k=k,
                        filter=filter_dict
                    )
                    all_docs_list.append(docs)
                    print(f'  ✅ 검색 완료: {len(docs)}개 문서 발견')
                except Exception as search_error:
                    print(f'  ❌ FAISS 검색 중 오류: {type(search_error).__name__}: {search_error}')
                    print(f'     검색어: "{term_str[:50]}..." (길이: {len(term_str)})')
                    all_docs_list.append([])
                    continue
            
            except Exception as e:
                print(f'  ❌ 검색 실패!')
                print(f'     오류 타입: {type(e).__name__}')
                print(f'     오류 메시지: {e}')
                print(f'     문제된 term: {repr(term)}')
                
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=20, max_execution_time=3000, handle_parsing_errors=True)

    return agent_executor


