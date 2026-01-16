# FINAL_RUNPOD_SERVER (SKN 17기 Final Project 2Team)

## 프로젝트 개요

### 프로젝트 명
말하는대로 (AI 기반 회의 자동화 시스템)

### 프로젝트 소개
음성에서 문서로. **실시간 녹음/업로드 → STT/화자 분리 → 도메인 기반 분석 → 안건/태스크 추출 → 캘린더 연동 → 웹 기반 회의록 생성**까지 **5단계 자동화**를 제공하는 회의 지원 솔루션입니다.

본 레포지토리는 **fast api 기반 모델 서버**만을 포함합니다.

- Django 서버 레포지토리: https://github.com/skn17-final-2team/final_django
- Django + 모델 서버 및 산출물 레포지토리: https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-FINAL-2Team

## 핵심 기능
| 구분 | 기능 | 설명 |
|:--|:--|:--|
| 음성 처리 | STT + 화자 분리 | 음성을 텍스트로 변환하고 발화자 구분 |
| 도메인 분석 | 도메인 용어 기반 컨텍스트 강화 | 사내/업무 용어를 인식하고 의미 보강(RAG) |
| 추출 | 안건/요약/태스크 자동 추출 | `누가/무엇을/언제까지`를 포함한 구조화 출력 |

## 기술 스택
| 분야 | 기술 |
|:---|:---|
| **Backend** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-4D9FEA?style=for-the-badge&logo=python&logoColor=white) |
| **AI Frameworks** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-8A2BE2?style=for-the-badge) |
| **AI Tech** | ![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?style=for-the-badge&logo=openai&logoColor=white) ![Pyannote.audio](https://img.shields.io/badge/Pyannote-audio-blue?style=for-the-badge) |
| **벡터 DB**| ![FAISS](https://img.shields.io/badge/FAISS-4A90E2?style=for-the-badge&logo=facebook&logoColor=white) |



## 프로젝트 구조
-   **main.py**: FastAPI 애플리케이션의 메인 파일. API 라우팅 및 요청 처리를 담당합니다.
-   **whisper_pannote.py**: Whisper와 Pyannote를 사용하여 오디오 처리 및 화자 분리를 수행합니다.
-   **sllm_tool_binding.py**: Qwen LLM 에이전트를 사용하여 회의록 텍스트를 분석하고 요약/태스크를 추출하는 핵심 로직을 포함합니다.
-   **main_model.py**: Hugging Face 모델과 토크나이저, FAISS DB를 로드하는 유틸리티 함수를 포함합니다.
-   **requirements.txt**: 프로젝트에 필요한 Python 패키지 목록입니다.
-   **prompts/**: LLM에 전달할 프롬프트 템플릿 (시스템, 요약, 태스크 추출)을 담고 있습니다.


# 1. 런팟 올리고 git clone
  ```bash
git clone https://github.com/skn17-final-2team/final_runpod_server.git

  ```

# 2. 환경 세팅
  ```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
cd final_runpod_server
pip install --no-cache-dir -r requirements.txt
  ```

# 3. Hugging Face 토큰 설정 본인 토큰으로 변경
  ```bash
echo "HF_TOKEN=<your_huggingface_token>" >> .env
  ```
# 4. uvicorn run
  ```bash
uvicorn main:app --host 0.0.0.0 --port 8000
  ```

# 4. 외부 접근 테스트
  ```http
https://<POD_ID>-8000.proxy.runpod.net/health
  ```
  - 아래와 같이 나오면 성공
  ```json
{ "status": "ok" }
  ```