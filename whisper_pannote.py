import os, requests, time
import tempfile
import torch
import whisper
import concurrent.futures

from pyannote.audio import Pipeline
from pyannote.audio.core import task as task_module
from dotenv import load_dotenv

torch.serialization.add_safe_globals([
    task_module.Specifications,
    task_module.Resolution,
    task_module.Problem,
])

# 토큰
load_dotenv()
token=os.getenv('HF_TOKEN')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pyannote(device):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    pipeline.to(torch.device(device))
    return pipeline

if torch.cuda.is_available():
    whisper_model = whisper.load_model("medium", device="cuda:0")
    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        pipeline = run_pyannote("cuda:0")
    else:
        pipeline = run_pyannote("cuda:1")
else:
    whisper_model = whisper.load_model("medium")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token)
    pipeline.to(device)

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_stt_diarization(audio_url, DEBUG=False):
    if not DEBUG:
        try:
            print('파일 다운로드 중')
            resp = requests.get(audio_url)
            if resp.status_code != 200:
                print("==== ERROR BODY ====")
                print(resp.text)
                return {"success": False, "message": f"Download failed ({resp.status_code})"}
            audio_bytes = resp.content
            print('파일 다운로드 완료')
        except Exception as e:
            return {"success": False, "error": {"type": "DownloadError", "message": str(e)}}

    else:
        with open(audio_url, "rb") as f:
            audio_bytes = f.read()
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        start = time.perf_counter()
        # Use globally loaded models
        print('위스퍼 + 파이아노트 병렬 실행 중')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            whisper_future = executor.submit(whisper_model.transcribe, tmp_path, language="ko")
            diarization_future = executor.submit(pipeline, tmp_path)

            whisper_result = whisper_future.result()
            diarization_result = diarization_future.result()

        annotation = diarization_result.speaker_diarization

        end = time.perf_counter()
        print(f"처리 시간: {end - start:.2f}초")

        final_segments = []

        for ws in whisper_result["segments"]:
            w_start, w_end = ws["start"], ws["end"]

            best_speaker = None
            best_overlap = 0.0

            # diarization matching
            for item in annotation.itertracks(yield_label=True):
                if len(item) == 2:
                    segment, speaker = item
                elif len(item) == 3:
                    segment, _, speaker = item
                else:
                    continue

                overlap = max(0, min(w_end, segment.end) - max(w_start, segment.start))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            final_segments.append({
                "speaker": best_speaker or "UNKNOWN",
                "start": float(w_start),
                "end": float(w_end),
                "text": ws["text"].strip()
            })

        # ------------------------
        # STEP 2: 연속 화자 merge
        # ------------------------
        merged_segments = []
        for seg in final_segments:
            if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"]:
                merged_segments[-1]["text"] += " " + seg["text"]
                merged_segments[-1]["end"] = seg["end"]  # end time 업데이트 (optional)
            else:
                merged_segments.append(seg)

        # formatted text 결과
        formatted_text = [{seg['speaker']: seg['text']} for seg in merged_segments]
        print('==================== 결과 ====================')
        print(formatted_text)
        return {
            "success": True,
            "data": {
                "full_text": formatted_text,
                "segments": merged_segments,
                "speakers": list({s['speaker'] for s in merged_segments}),
                "raw_transcript": whisper_result["text"]
            }
        }

    except Exception as e:
        print(f"An error occurred during audio processing: {e}")
        return {
            "success": False,
            "error": {
                "type": "AudioProcessingError",
                "message": "Error during audio processing",
                "detail": str(e)
            }
        }
    finally:
        # Ensure temporary file is deleted
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Temporary file {tmp_path} deleted.")

if __name__ == "__main__":
    result = run_stt_diarization('test.wav', DEBUG=True)
    print(result)