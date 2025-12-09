import os, time, torch, platform, re, json 
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path

from peft import PeftModel
from langchain.vectorstores import FAISS
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").strip()
PROMPTS = {
    "summarizer": (PROMPT_DIR / "summarizer.txt").read_text(encoding="utf-8").strip(),
    "task_extractor": (PROMPT_DIR / "extract_tasks.txt").read_text(encoding="utf-8").strip(),
}

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
ft_model_name = "CHOROROK/Qwen2.5_1.5B_trained_model_v3"


# ===== 이스케이트 ====
def escape_curly(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


# ===== 벡터 DB 로드 =====
def load_faiss_db(db_path: str):
    embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("🔵 FAISS DB 로드 완료!\n")
    return vector_store, embedding_model


# ===== 모델 로드 =====
def load_model_q(model_name, adapter_name: str | None = None):
    if platform.system() == "Windows":
        print("⚠ Windows에서는 4bit 불가 → FP16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        print("🔵 Linux/RunPod 환경: 4bit 없이 bf16로 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,   # 안 되면 torch.float16 로 바꿔도 됨
            device_map="auto",
        )

    if adapter_name:
        print(f"🔵 LoRA/PEFT 어댑터 로드: {adapter_name}")
        model = PeftModel.from_pretrained(base_model, adapter_name)
    else:
        model = base_model

    text_gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, 
        max_new_tokens = 2048, 
        temperature=0.2,
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipe)
    return llm


# ===== 도메인 필터 =====
def make_filter(filter: dict):
    if any(filter.values()):
        main_filter = filter.copy()
    else:
        main_filter = None
    return main_filter