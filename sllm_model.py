import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
from peft import PeftModel
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv('HF_TOKEN'))

with open('prompts/summarizer.txt', 'r', encoding='utf-8') as f:
    SUMMARIZER_PROMPT = f.read()
with open('prompts/summarizer.txt', 'r', encoding='utf-8') as f:
    SUMMARIZER_PROMPT = f.read()
with open('prompts/summarizer.txt', 'r', encoding='utf-8') as f:
    SUMMARIZER_PROMPT = f.read()

def summarize(transcript):
    return None

def extract_tasks(transcript):
    return None

def extract_issues(transcript):
    return None

def run_inference_model(transcript: str):
    summary = summarize(transcript)
    tasks = extract_tasks(transcript)
    issues = extract_issues(transcript)

    return {
            "success": True,
            "data": {
                "summary": summary,
                "tasks": tasks,
                "issues": issues
            }
        }

if __name__ == "__main__":
    # q = input('전문: ')
    print(SUMMARIZER_PROMPT[:100])