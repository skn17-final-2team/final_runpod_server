from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from whisper_pannote import run_stt_diarization
from sllm_model import process_transcript_with_chunks
from sllm_tool_binding import agent_main

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

class AudioRequest(BaseModel):
    audio_url: str

class InferenceRequest(BaseModel):
    transcript: str
    domain: list[str] = Field(default_factory=list)

# 포트연결 정상 적동 되었는지 확인용
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": __import__('time').time()}

@app.post("/stt")
async def stt(req: AudioRequest):
    if not req.audio_url:
        raise HTTPException(status_code=422, detail="audio_url required")
    try:
        result = run_stt_diarization(req.audio_url)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})

@app.post("/inference")
async def inference(req: InferenceRequest):
    if not req.transcript:
        raise HTTPException(status_code=422, detail="transcript required")
    try:
        # result = process_transcript_with_chunks(req.transcript, req.domain)
        result = agent_main(req.transcript, req.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})
