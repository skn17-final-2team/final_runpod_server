import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
from peft import PeftModel
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv('HF_TOKEN'))



if __name__ == "__main__":
    q = input('질문: ')