from sllm_model import build_agent
from main_model import load_model_q
from main_model import load_faiss_db
from main_model import escape_curly

model = load_model_q("Qwen/Qwen2.5-1.5B-Instruct")
# model = load_model_q("CHOROROK/Qwen2.5_1.5B_trained_model_v3")
db_path = './faiss_db_merged'
vector_store, embedding_model = load_faiss_db(db_path)


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