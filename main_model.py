import os, torch, platform, json 
from dotenv import load_dotenv
from huggingface_hub import login

from peft import PeftModel
from langchain.vectorstores import FAISS
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ===== 모델 설정 =====
base_model_name = "Qwen/Qwen3-8B"
ft_model_name = ""

# # ===== 이스케이트 =====
# def escape_curly(text: str) -> str:
#     return text.replace("{", "{{").replace("}", "}}")

# ===== 전문 전처리 (JSON -> 텍스트) =====
def preprocess_transcript(transcript):
    # 1) 이미 파이썬 객체(list/dict)로 들어온 경우
    if isinstance(transcript, list):
        data = transcript
    elif isinstance(transcript, dict):
        data = [transcript]
    else:
        # 2) 문자열이면 JSON 파싱 시도
        try:
            data = json.loads(transcript)
        except Exception:
            return str(transcript)

    lines = []
    for entry in data:
        if isinstance(entry, dict):
            for speaker, text in entry.items():
                lines.append(f"{speaker}: {text}")
        else:
            lines.append(str(entry))

    return "\n\n".join(lines)
        
# ===== 벡터DB 로드 =====
def load_faiss_db(db_path: str):
    # 임베딩 모델 로드 (최대 시퀀스 길이 제한 고려)
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        encode_kwargs={'normalize_embeddings': True}  # 정규화로 검색 품질 향상
    )
    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("🔵 FAISS DB 로드 완료!\n")
    return vector_store, embedding_model


# ===== 형식지정 =====
class HFTextGenLLM(LLM):
    """HF text-generation pipeline을 감싸는, 비-스트리밍 LLM 래퍼."""
    pipe: Any
    tokenizer: Any = None
    model: Any = None
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "hf_text_generation_pipeline"

    def _normalize_prompt(self, prompt) -> str:
        """str로 정규화하고 유니코드 문제 해결."""
        result = ""

        if isinstance(prompt, PromptValue): # prompt template > string
            result = prompt.to_string()
        elif isinstance(prompt, list) and prompt:  # chatmessage list > content만 추출
            if isinstance(prompt[0], BaseMessage):
                result = "\n".join(m.content for m in prompt)
            else:
                result = str(prompt)
        elif isinstance(prompt, str): # string 
            result = prompt
        else:
            result = str(prompt)

        # 서로게이트 문자 제거 (여기서 먼저 처리) : 메모장 깨지는 거 처리 
        try:
            result = result.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='ignore')
        except:
            # 폴백: 서로게이트를 무시하고 재인코딩
            result = result.encode('utf-16', 'surrogatepass').decode('utf-16', errors='ignore')
        return result


    def _call(self, prompt, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        text = self._normalize_prompt(prompt)

        if not isinstance(text, str): # str 변환
            print(f"⚠️ 경고: 프롬프트가 문자열이 아닙니다. 타입: {type(text)}")
            text = str(text)

        print(f"🔹 LLM 입력 프롬프트 길이: {len(text)} 글자")
        text = str(text).strip()         # 문자열 타입 강제 확인 및 정리 & 공백제거

        # 너무 긴 입력은 잘라내기 (토큰 기준 약 6000개) : 안전장치 (앞에 내용만 반환)
        MAX_INPUT_CHARS = 12000
        if len(text) > MAX_INPUT_CHARS:
            print(f"⚠️ 입력이 너무 깁니다 ({len(text)}자). {MAX_INPUT_CHARS}자로 자릅니다.")
            text = text[:MAX_INPUT_CHARS]

        # HF text-generation pipeline 실행
        try:
            if self.tokenizer and self.model: # qwen chat template
                print("🔹 직접 tokenizer/model 사용 모드")
                import torch
                try:
                    print(f"   Tokenizer 타입: {type(self.tokenizer)}")
                    messages = [{"role": "user", "content": text}]

                    if hasattr(self.tokenizer, 'apply_chat_template'):  # chat template
                        print("   apply_chat_template 사용")
                        formatted_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        print("   chat template 없음, 직접 사용")  # 직접 생성
                        formatted_text = text

                    # 토크나이즈
                    inputs = self.tokenizer(
                        formatted_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=6144
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    print(f"   토크나이즈 성공: inputs shape = {inputs['input_ids'].shape}")


                except Exception as tok_error:
                    print(f"   ❌ Tokenizer 호출 실패: {tok_error}")
                    print(f"   폴백: pipeline 사용")
                    raise  # 폴백하도록 에러 발생시킴

                # 생성
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=True
                    )

                # 디코딩 (새로 생성된 것만)
                generated_text = self.tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                return generated_text
            else:
                outputs = self.pipe(text)  # tokenizer/model 직접 사용 불가 시, 기존 pipeline 사용.
                if not outputs:
                    return ""

                first = outputs[0]
                generated = first.get("generated_text") or first.get("text") or ""

                if stop:
                    for s in stop:
                        if s in generated:
                            generated = generated.split(s)[0]
                            break

                return generated

        except Exception as e:
            print(f"❌ Pipeline 실행 중 에러 발생!")
            print(f"   에러 타입: {type(e).__name__}")
            print(f"   에러 메시지: {e}")
            print(f"   입력 타입: {type(text)}")
            print(f"   입력 길이: {len(text) if isinstance(text, str) else 'N/A'}")
            print(f"   입력 샘플: {text[:200] if isinstance(text, str) else text}...")

            # 추가 디버깅: 입력의 repr 확인
            print(f"   입력 repr: {repr(text[:100])}")
            raise



# ===== 모델 로드 =====
def load_model_q(model_name: str | None = base_model_name , adapter_name: str | None = ft_model_name):

    # Fast tokenizer 비활성화 (TextEncodeInput 에러 방지)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("🔹 Slow tokenizer 사용")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("🔹 Fast tokenizer 사용 (폴백)")

    # 긴 입력 처리를 위한 설정
    if tokenizer.model_max_length > 100000:  # 비정상적으로 큰 값인 경우
        tokenizer.model_max_length = 8192    # 합리적인 값으로 설정
    print(f"🔹 Tokenizer max_length: {tokenizer.model_max_length}")
    
    if platform.system() == "Windows":
        print("⚠ Windows에서는 4bit 불가 → FP16로 로드합니다.")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        print("🔵 Linux/RunPod 환경: 4bit 없이 bf16로 로드합니다.")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   
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
        max_new_tokens=2048,
        temperature=0.2,
        top_p=0.9
    )

    llm = HFTextGenLLM(pipe=text_gen_pipe, tokenizer=tokenizer, model=model)
    return llm

