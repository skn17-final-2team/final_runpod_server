# FINAL_RUNPOD_SERVER

# 1. 런팟 올리고 git clone
  ```bash
    git clone https://github.com/skn17-final-2team/final_runpod_server.git
  ```

# 2. 환경 세팅
  ```bash
    cd final_runpod_server
    echo "HF_TOKEN=<your_huggingface_token>" >> .env
    pip install --no-cache-dir -r requirements.txt
  ```
# 3. uvicorn run
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