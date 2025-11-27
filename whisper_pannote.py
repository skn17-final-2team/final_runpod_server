import os, requests
import tempfile
import torch
import whisper
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# 토큰
load_dotenv()
token=os.getenv('HF_TOKEN')

whisper_model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token).to(torch.device("cuda"))

def run_stt_diarization(audio_url):
    resp = requests.get(audio_url)
    print("status:", resp.status_code)

    if resp.status_code != 200:
        print("==== ERROR BODY ====")
        print(resp.text)
        return {
            "success": False,
            "message": resp.text
        }

    audio_bytes = resp.content

    tmp_path = None # Initialize tmp_path
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Use globally loaded models
        whisper_result = whisper_model.transcribe(tmp_path, language="ko")
        diarization_result = pipeline(tmp_path)
        annotation = diarization_result.speaker_diarization

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
        formatted_text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in merged_segments)

        return {
                "success": True,
                "message": {
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
            "message": f"Error during audio processing: {e}"
        }
    finally:
        # Ensure temporary file is deleted
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Temporary file {tmp_path} deleted.")