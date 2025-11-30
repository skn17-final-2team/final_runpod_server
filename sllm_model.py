import os, json, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
from peft import PeftModel
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").strip()
PROMPTS = {
    "summarizer": (PROMPT_DIR / "summarizer.txt").read_text(encoding="utf-8").strip(),
    "task_extractor": (PROMPT_DIR / "extract_tasks.txt").read_text(encoding="utf-8").strip(),
    "issue_extractor": (PROMPT_DIR / "extract_issues.txt").read_text(encoding="utf-8").strip(),
}

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
peft_path = "poketmon/Qwen2.5_1.5B_trained_model_v2"

tokenizer = AutoTokenizer.from_pretrained(
    peft_path,
    trust_remote_code=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, peft_path)
pipe = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    temperature=0.3,
    top_p=0.9,
)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

def extract_json(text: str) -> dict | None:
    if match := re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE):
        try:
            return json.loads(match.group(1))
        except:
            pass

    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except:
                    break
    return None

def run_inference_model(transcript: str):
    return {
            "success": True,
            "data": {
                "summary": None,
                "tasks": None,
                "issues": None
            }
        }

if __name__ == "__main__":
    # q = input('전문: ')
    print(SYSTEM_PROMPT[:100])
    print("=========")
    while True:
        messages = [
            {"role": "system", "content": "모든 질문에 대해 반드시 한국어로 대답하시오."},
            {"role": "user", "content": input("질문: ")},
        ]
        response = pipe(messages, max_new_tokens=1024)
        print(response[0]['generated_text'])