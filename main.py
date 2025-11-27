from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from whisper_pannote import run_stt_diarization

app = FastAPI(title="Runpod STT + Pyannote Server")

# CORS 정책 때문에 응답 못 받는 거 해결용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (테스트용)
    # allow_origins=["*"],  # 배포된 IP:PORT 또는 도메인 넣기
    allow_credentials=True,
    # allow_methods=["*"],
    allow_methods=["GET", "POST"],  # 필요한 메서드만 허용
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

# 포트연결 정상 적동 되었는지 확인용
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/stt")
async def stt(req: Query):
    return run_stt_diarization(req.text)