import os, time, torch, platform, re, json 
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path

from peft import PeftModel
from langchain.vectorstores import FAISS
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage


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


# ===== 형식지정 =====
class HFTextGenLLM(LLM):
    """HF text-generation pipeline을 감싸는, 비-스트리밍 LLM 래퍼."""

    # 🔹 BaseModel(LLM)에 등록될 필드
    pipe: Any

    # 🔹 HF pipeline 같은 임의 타입을 필드로 허용하려면 꼭 필요
    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def _llm_type(self) -> str:
        return "hf_text_generation_pipeline"

    def _normalize_prompt(self, prompt) -> str:
        """LangChain 쪽에서 넘어오는 다양한 타입을 항상 str로 정규화."""
        # 1) PromptValue (PromptTemplate | ChatPromptTemplate 결과)
        if isinstance(prompt, PromptValue):
            return prompt.to_string()

        # 2) 메시지 리스트
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[0], BaseMessage):
                return "\n".join(m.content for m in prompt)

        # 3) 이미 문자열이면 그대로
        if isinstance(prompt, str):
            return prompt

        # 4) 나머지는 그냥 문자열 캐스팅
        return str(prompt)

    def _call(
        self,
        prompt,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """LangChain에서 호출하는 메인 엔트리. 내부적으로 HF pipeline 한 번 실행."""
        text = self._normalize_prompt(prompt)

        # HF text-generation pipeline 실행
        outputs = self.pipe(text)

        # transformers pipeline("text-generation") 기본 반환 형식: [{"generated_text": "..."}]
        if not outputs:
            return ""

        first = outputs[0]
        generated = first.get("generated_text") or first.get("text") or ""

        # stop 토큰이 있으면 거기서 잘라주기
        if stop:
            for s in stop:
                if s in generated:
                    generated = generated.split(s)[0]
                    break

        return generated


# ===== 모델 로드 =====
def load_model_q(model_name: str , adapter_name: str | None = ft_model_name):
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

    llm = HFTextGenLLM(pipe = text_gen_pipe)
    return llm


# ===== 도메인 필터 =====
def make_filter(filter: dict):
    if any(filter.values()):
        main_filter = filter.copy()
    else:
        main_filter = None
    return main_filter