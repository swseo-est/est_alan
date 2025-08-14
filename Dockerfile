FROM docker.io/langchain/langgraph-api:3.11

# (필수) pg_isready 제공 패키지
USER root
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client \
 && rm -rf /var/lib/apt/lists/*

# --- 로컬 패키지 주입 ---
# 빌드 컨텍스트가 프로젝트 루트(.) 라는 가정
# (compose에서 context: . , dockerfile: api/Dockerfile)
ADD src /deps/est_alan

# --- 로컬 의존성 설치 ---
# /api/constraints.txt 를 최대한 존중해서 설치
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e /deps/*

# --- LangGraph 체크포인터(+psycopg3) 설치 ---
# constraints에 없을 수 있으므로 별도 단계에서 설치
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir \
    langgraph-checkpoint-postgres \
    "psycopg[binary]"

# --- langgraph-api 재설치(원 이미지 레이아웃 유지) ---
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license \
 && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

# -- Copy Local files
COPY entrypoint.sh /entrypoint.sh
COPY graph.json /graph.json
COPY est-alan-dev-account.json /est-alan-dev-account.json

RUN chmod +x /entrypoint.sh

WORKDIR /deps/est_alan
ENTRYPOINT ["/entrypoint.sh"]
