# Python 3.11 버전 베이스 이미지 사용
FROM python:3.11

# 환경 설정 (bytecode / 버퍼)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 지정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 나머지 코드 전체 복사
COPY . .

# 포트 명시 (FastAPI가 8000에서 실행됨)
EXPOSE 8000

# 실행 명령 (Runpod이 컨테이너 시작 시 실행할 것)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]