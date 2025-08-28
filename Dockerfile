FROM docker.io/langchain/langgraph-api:3.11

# --- 1. 작업 디렉토리 준비 ---
WORKDIR /deps/est_alan

# --- 2. pyproject.toml + uv.lock 먼저 복사 ---
#     -> 의존성 설치 레이어 캐시 활용
COPY src/pyproject.toml src/uv.lock /deps/est_alan/

# --- 3. 의존성 설치 (lockfile 기준) ---
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt /deps/est_alan

# --- 4. src 전체 복사 (항상 최신 반영) ---
COPY src /deps/est_alan

# --- 5. 개발 모드 설치 (코드 바뀌면 여기부터 새 빌드) ---
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -e /deps/*

# --- 6. pyproject 기준으로 sync (의존성 drift 방지) ---
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps /deps/est_alan

# --- 7. langgraph-api 보호 레이어 ---
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \
    touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

# --- 8. 불필요한 빌드 도구 제거 ---
RUN pip uninstall -y pip setuptools wheel && \
    rm -rf /usr/local/lib/python*/site-packages/pip* \
           /usr/local/lib/python*/site-packages/setuptools* \
           /usr/local/lib/python*/site-packages/wheel* && \
    find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* \
           /usr/lib/python*/site-packages/setuptools* \
           /usr/lib/python*/site-packages/wheel* && \
    find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel

# --- 9. 로컬 파일 추가 ---
COPY entrypoint.sh /entrypoint.sh
COPY graph.json /graph.json
COPY est-alan-dev-account.json /deps/est_alan/est-alan-dev-account.json
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
