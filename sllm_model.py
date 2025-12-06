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
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate

# from retrieval import retrievel
from main_model import load_model_q
from main_model import load_faiss_db
from main_model import escape_curly


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

model = load_model_q("CHOROROK/Qwen2.5_1.5B_trained_model_v3")
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

def build_agent(model, vector_store) :

    safe_summarizer = escape_curly(PROMPTS["summarizer"])
    safe_task_prompt = escape_curly(PROMPTS["task_extractor"])

    prompt = PromptTemplate.from_template('''
        You are an AI meeting-analysis agent specialized in IT projects and software development.
        You will receive user requests and (often) a meeting transcript about IT topics
        (e.g., architecture, infra, APIs, CI/CD, data, AI/ML, product decisions).

        You must answer as accurately as possible using the available tools.
        You have access to the following tools:
        {tools}

        Your primary goals when handling a meeting-related request are:

        1) Understand the meeting context:
           - 목적(purpose), 참여자(participants), 결정사항(decisions), 미해결 이슈(open issues)를 파악한다.
        2) When necessary, clarify or look up IT/technical/domain terms or concepts using tools.
        3) From the meeting transcript, you must be able to:
           - extract issue list (이슈 목록 추출)
           - summarize the meeting (회의 요약)
           - extract follow-up tasks (후속 태스크)
        4) When you extract issue list, summary, decisions, or tasks,
           you MUST follow the dedicated JSON prompts described below
           (Summary JSON Prompt, Tasks JSON Prompt).
        5) Ground your answers in the meeting transcript and retrieved domain documents; 
           NEVER hallucinate requirements or decisions that are not supported by the content.


        [Summary JSON Prompt – for structured agenda/summary JSON]
        When the user explicitly requests a JSON-formatted summary/agendas
        (or an internal step requires JSON summary), you MUST conceptually apply
        the following instructions:
        - Input: the full meeting transcript (denoted as {{transcript}}).
        - You must output a SINGLE valid JSON object with fields:
          - "id": "{{record_id}}"  (use the id provided by the system)
          - "summary": an object containing agenda_1, agenda_2, ... (5W3H 구조)
          - "agendas": a list of agenda items with title/description
        - "summary" object:
          - Contains keys: "agenda_1", "agenda_2", ...
          - Each agenda_n must contain the following 5W3H fields:
            - "who", "what", "when", "where", "why",
              "how", "how_much", "how_long"
          - Every field MUST be filled ONLY with words/phrases
            that appear in the original transcript.
          - If a certain 5W3H item is not explicitly stated in the transcript,
            set that field to null.
          - Do NOT invent or infer new facts that are not present in the transcript.
        - "agendas" list:
          - Each element is an object: {{ "title": "...", "description": "..." }}
          - If there are no clear agendas, use [].
          - "description" can be null if there is no description.
          - The order of items in "agendas" must match the order of agenda_1, agenda_2, ... in "summary".
        - Global rules for the Summary JSON:
          1. Output exactly ONE valid JSON object (no extra text, no markdown).
          2. Use the {{record_id}} exactly as provided by the system.
          3. Use only information that appears in the transcript.
          4. Do NOT hallucinate or add external assumptions.
          5. When this Summary JSON Prompt is used, the output MUST be only JSON
             and nothing else.

        [Tasks JSON Prompt – for structured tasks JSON]
        When the user explicitly requests a JSON-formatted tasks list
        (or an internal step requires JSON tasks), you MUST conceptually apply
        the following instructions:
        - Input: the full meeting transcript (denoted as {input}).
        - You must output a SINGLE valid JSON object with fields:
          - "id": "{{record_id}}"  (use the id provided by the system)
          - "tasks": a list of task objects
        - Each task object has:
          - "description": 태스크 내용 (해야 할 일)
          - "assignee": 담당자 이름 또는 역할 (없으면 null 또는 적절히 비워둘 수 있음)
          - "due": 전문에 등장하는 마감일 표현 그대로 적는다
                   (예: "오늘 오후", "다음 주 초", "이번 주 금요일")
                   없으면 "*(언급 없음)*" 으로 표기
          - "due_date": ISO 형식 날짜 "YYYY-MM-DD"
                        - {{current_date}} 기준으로 계산 가능할 때만 날짜를 채운다.
                        - 계산 불가하거나 due 정보가 없으면 null.
        - 전체 "tasks" 리스트:
          - 태스크가 여러 개면 모두 뽑는다 (개수 제한 없음).
          - 태스크가 하나도 없으면 "tasks": [].
        - Global rules for the Tasks JSON:
          1. Output exactly ONE valid JSON object (no extra text, no markdown).
          2. Use the {{record_id}} exactly as provided by the system.
          3. Do NOT limit tasks to IT-only; any actionable item in the transcript
             (업무, 준비사항, 연락, 공유 등) 은 모두 태스크로 추출한다.
          4. Do NOT hallucinate; only use information explicitly mentioned
             in the transcript.
          5. When this Tasks JSON Prompt is used, the output MUST be only JSON
             and nothing else.

        [How to use these prompts in ReAct]
        - In your ReAct reasoning, when you need a structured JSON result,
          you should think (in Thought) about using the appropriate JSON prompt:
          - Use the Summary JSON Prompt when you need structured summary/agendas.
          - Use the Tasks JSON Prompt when you need structured tasks.
        - However, in normal conversational answers (Final Answer),
          you can respond in natural Korean, summarizing, listing issues,
          decisions, and tasks in a human-readable way.
          Only when the user explicitly asks for JSON output, or the system
          requires it, you must follow the JSON prompts strictly.


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
        - If the user request is general chit-chat, a simple greeting, or a very simple question,
          you MAY skip Action/Action Input/Observation and respond directly with Final Answer.
        - If you need additional domain knowledge or definitions,
          choose the most appropriate tool from [{tool_names}] and use it.
        - Use the meeting transcript and retrieved documents as the primary source of truth.
        - When you summarize or extract issues/decisions/tasks, be faithful to the transcript.
        - Final Answer MUST be written in Korean, unless the user clearly asks for another language.

        Begin!

        transcript:{input}
        Thought:{agent_scratchpad}

        '''
    )

    @tool
    def retrieval(term_list) :
        """FAISS 벡터스토어에서 전문 중 모르는 단어를 검색해 단어 정의를 반환하는 툴."""
        print('여기다 여기!!!', term_list)      
        for term in term_list:
            docs = self.retriever.invoke(term)
            if not docs:
                # 못 찾은 용어는 패스 
                print(f'@#$@#$@${term}에 대한 용어 못찾음@#$#@$@#')
                continue        
            # 가장 관련도 높은 문서 1~2개를 합쳐서 정의로 사용
            defs = []
            for d in docs[:2]:
                ans = d.metadata.get("answer") or d.page_content
                defs.append(ans.strip())        
            definitions[term] = "\n\n".join(defs)
            print('여기다 여기!!!', definitions)
            retrieval_result = {"output": json.dumps({"definitions": definitions}, ensure_ascii=False)}     
            # 3) JSON 문자열로 반환
        return retrieval_result

    tools = [retrieval]

    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20, max_execution_time=400, handle_parsing_errors=True)

    return agent_executor


if __name__ == "__main__":
    print("회의록 전문을 입력하세요! 종료하려면 'exit' 입력\n")
    agent = build_agent(model=model, vector_store=vector_store)

    while True:
        query = input("전문: ")
        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        agent_result = agent.invoke({"input": query})
        result_text = agent_result["output"]

        # AgentExecutor는 보통 {"output": "...", ...} 형태 반환
        print("\n모델 응답(JSON):\n", result_text)
            # result_final = {"success": True, "data": {"summary": result_text['agedas'], "tasks": result_text['tasks']}}
            # print(result_final)
