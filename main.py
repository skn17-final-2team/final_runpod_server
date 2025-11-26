from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Runpod STT + Pyannote + sLLM Server")

# CORS 정책 때문에 응답 못 받는 거 해결용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (테스트용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 포트연결 정상 적동 되었는지 확인용
@app.get("/health")
def health_check():
    return {"status": "ok"}