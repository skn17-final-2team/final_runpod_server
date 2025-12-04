import os, time, torch, platform, re, json 
from dotenv import load_dotenv
from huggingface_hub import login

from langchain.vectorstores import FAISS
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import tool
from langchain.agents import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import MessagesPlaceholder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# local_path = "/workspace/snowflake-arctic-embed-l-v2.0-ko"

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer(local_path)

# ===== 벡터 DB 로드 =====
def load_faiss_db(db_path: str):
    embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("🔵 FAISS DB 로드 완료!\n")
    return vector_store, embedding_model


# ===== 모델 로드 =====
def load_model_q(model_name):
    if platform.system() == "Windows":
        print("⚠ Windows에서는 4bit 불가 → FP16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        print("🔵 Linux/RunPod 환경: 4bit 없이 bf16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # 안 되면 torch.float16 로 바꿔도 됨
            device_map="auto",
        )

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
    chat_llm = ChatHuggingFace(llm=llm)
    return chat_llm

# agent 생성 
def build_agent(llm, vector_store, default_domain="IT"):

    system_prompt = ChatPromptTemplate.from_template("""
        You are an AI meeting-analysis agent specialized in IT projects and software development.
        You will receive user requests and (often) a meeting transcript about IT topics
        (e.g., architecture, infra, APIs, CI/CD, data, AI/ML, product decisions).
        
        You must answer as accurately as possible using the available tools.
        
        You have access to the following tools:
        
        {tools}
        
        Your goals when handling a meeting-related request are:
        1) Understand the meeting context (purpose, participants, decisions, open issues).
        2) When necessary, clarify or look up IT/technical terms or concepts using tools.
        3) When the user asks, summarize the meeting, extract decisions, action items, risks, or follow-up tasks.
        4) Ground your answers in the meeting transcript and retrieved IT-domain documents; avoid hallucinating
           requirements or decisions that are not supported by the content.
        
        Use the following ReAct-style format:
        
        Question: the input question or request you must answer
        Thought: you should always think about what to do next
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question in Korean
        
        Important rules:
        - If the user request is general chit-chat, a simple greeting, or a very simple question,
          you MAY skip Action/Action Input/Observation and respond directly with Final Answer.
        - If you need additional IT knowledge, definitions, or related internal documents,
          choose the most appropriate tool from [{tool_names}] and use it.
        - Use the meeting transcript and retrieved documents as the primary source of truth.
        - When you summarize or extract tasks/decisions, be faithful to the transcript.
        - Final Answer MUST be written in Korean, unless the user clearly asks for another language.
        
        Begin!
        
        Question: {query}
        Thought:{agent_scratchpad}
        """)

    tools = []
    tools.append(
        Tool(
            name="lookup_definition",
            func=lambda q: lookup_definition(q, vector_store, default_filter),
            description="회의록 전문에 모호한 단어가 포함됐을 때, 문서의 단어 정의를 참고해서 요약 및 태스크 추출",
            return_direct=False
        )
    )

    # agent = create_tool_calling_agent(model, tools, prompt)
    agent = create_react_agent(llm, tools, system_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=30,
        max_execution_time=60,
        handle_parsing_errors=True,
    )

    return agent_executor

# ===== 도메인 필터 =====
def make_filter(filter: dict):
    if any(filter.values()):
        main_filter = filter.copy()
    else:
        main_filter = None
    return main_filter

# 정의 반환
@tool("lookup_definition")
def lookup_definition(terms: str) -> str:
    """IT 용어나 회의에서 등장한 모호한 단어를 벡터DB에서 검색하여 정의를 반환한다."""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.8,
            "filter": {"domain": default_domain},
        },
    )
    term_list = [t.strip() for t in re.split(r"[,;\n]+", terms) if t.strip()]
    lines = []
    for term in term_list:
        docs = retriever.invoke(term)
        if not docs:
            return f"'{term}'에 대한 정의를 찾을 수 없습니다."
        
        defs = []
        for d in docs[:3]:
            ans = d.metadata.get("answer") or d.page_content
            defs.append(ans.strip())
        lines.append(f"{term}:\n" + "\n\n".join(defs))
    # print('🐋 모르는 단어:', term_list)
    return "\n\n---\n\n".join(lines), term_list


def make_chain(model):
    instruction = """
    당신은 회의록 전문을 분석하는 AI 비서입니다.
    아래 지침을 따라 JSON 하나만 생성하세요.

    1) 회의 요약(summary)
       - 3~6문장 정도의 한국어 문단 또는 5~10개 불릿으로,
         누가 어떤 결정을 했고, 어떤 기준/지표/제약이 논의되었는지 구체적으로 작성
       - 가능하면 PM/PO/QA/FE/BE/AI 등 역할별로 핵심 발언을 정리

    2) 태스크(tasks)
       - 회의에서 실제로 "해야 할 일"로 들리는 내용을 최대한 잘게 쪼개서 추출
       - 각 태스크는 아래 필드를 가진 객체:
         - owner: 담당자 이름 또는 역할 (예: "박지은(PM)", "김현우(PO)")
         - task: 구체적인 행동 문장 (예: "MVP Must/Should/Won't 리스트 문서화")
         - due: 기한. 회의에서 명시됐으면 구체 날짜, 없으면 "TBD" 또는 "" 사용

    3) definitions 활용
       - {rag_result_text} 에는 회의에서 사용된 용어에 대한 정의 JSON이 들어있다고 가정
       - 해당 용어들이 등장하면, 그 맥락을 이해하는 데 참고만 하고,
         summary / tasks 안에 불필요하게 그대로 복붙하지는 마세요.

    출력 형식:
    - 반드시 아래 JSON 스키마 한 개만 반환하세요.
    - keys를 중복 정의하지 마세요. (예: "tasks"를 두 번 쓰지 말 것)

    {{
      "summary": "<자세한 한국어 요약>",
      "tasks": [
        {{
          "who": "이름 또는 역할",
          "what": "해야 할 일",
          "when": "YYYY-MM-DD 또는 'TBD' 혹은 빈 문자열"
        }},
        ...
      ]
    }}
    """

    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction),
        HumanMessagePromptTemplate.from_template(
            "사용자 회의록: {text}\n\n"
            "참고 문서:\n{rag_result_text}\n\n"
            "위의 지침을 준수하여 오직 사용자가 입력하는 회의록에 대한 답변만 생성해야 해."
        )
    ])
    chain = prompt | model | parser
    return chain

# ===== 메인 =====
if __name__ == "__main__":
    load_dotenv()
    db_path = './faiss_db/rag_it_tta'
    HF_TOKEN = os.getenv('HF_TOKEN')
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device set to:", device)

    vector_store, embedding_model = load_faiss_db(db_path)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = load_model_q(model_name)

    ft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    ft_model = load_model_q(ft_model_name) 

    # 에이전트 생성
    agent = build_agent(model, vector_store, default_domain='IT')

    # 파튜 모델
    chain = make_chain(ft_model)
    print("회의록 전문을 입력하세요! 종료하려면 'exit' 입력\n")

    while True:
        query = input("전문: ")
        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        # 👉 에이전트에게 그냥 통으로 던진다?????????
        # result = agent.invoke({"input": query})
        # result = agent.invoke({"messages": [{"role": "user", "input": query}]})

        print("\n--- 🔍 에이전트: 단어 정의 추출중 ---")
        agent_result = agent.invoke({"query": query})
        rag_result_text = agent_result["output"]
        print(" 🔍 에이전트 definitions:", rag_result_text)

        print("\n--- 🤖 파튜 모델: 요약 생성중 ---")
        result = chain.invoke({
            "text": query,
            "rag_result_text": rag_result_text
        })

        # AgentExecutor는 보통 {"output": "...", ...} 형태 반환
        print("\n모델 응답(JSON):\n", result)