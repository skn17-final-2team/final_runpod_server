from sllm_model import build_agent, process_transcript_with_chunks
from main_model import load_model_q
from main_model import load_faiss_db
from pathlib import Path
import json

db_path = "./faiss_db_merged"
vector_store, embedding_model = load_faiss_db(db_path)

if __name__ == "__main__":
    # 도메인 입력 (django랑 연결해야함)
    domain = input("도메인 입력 (accounting, design, marketing_economy, it): ").strip()

    # 모델 연결 (1.5b 파튜 기본값 설정됨)
    model = load_model_q()

    # 테스트용 루프 (django 연결 시, 필요 x / 현재 exit 입력 전까지 계속 반복 중(1회로 변경 필요))
    while True:
        print("\n" + "="*60)
        print("회의록 전문을 입력하세요!")
        print("- 긴 전문은 자동으로 청크로 나눠서 처리됩니다")
        print("- 전체 전문 기반으로 안건/요약/태스크를 추출합니다")
        print("- 종료하려면 'exit' 입력")
        print("="*60 + "\n")

        query = input("전문: ")
        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        # 청크 처리 및 전체 요약/태스크 추출
        result = process_transcript_with_chunks(transcript=query, domain=domain)

        # 결과 출력
        print("\n" + "="*60)
        print("최종 결과")
        print("="*60 + "\n")
        print("📝 안건/요약:")
        print("-" * 60)
        if isinstance(result["full_summary"], dict) and "error" in result["full_summary"]:
            print(f"❌ 에러: {result['full_summary']['error']}")
        else:
            print(result["full_summary"])

        print("\n📋 태스크:")
        print("-" * 60)
        if isinstance(result["full_tasks"], dict) and "error" in result["full_tasks"]:
            print(f"❌ 에러: {result['full_tasks']['error']}")
        else:
            print(result["full_tasks"])

        print("\n" + "="*60 + "\n")

        # JSON 형식으로도 출력 (최종 결과)
        try:
            result_json = json.dumps(result, ensure_ascii=False, indent=2)
            print("\n JSON 결과 :")
            print(result_json)
        except:
            pass