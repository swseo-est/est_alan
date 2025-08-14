#!/usr/bin/env sh
set -eu

# --- LangServe 그래프 로드(선택) ---
if [ -f /graph.json ]; then
  export LANGSERVE_GRAPHS="$(cat /graph.json)"
fi

# --- Postgres Ready 대기 ---
if [ -n "${POSTGRES_URI:-}" ]; then
  echo "⏳ Waiting for Postgres to be ready..."
  TIMEOUT="${POSTGRES_READY_TIMEOUT:-120}"   # seconds
  START="$(date +%s)"
  until pg_isready -d "$POSTGRES_URI" >/dev/null 2>&1; do
    NOW="$(date +%s)"
    if [ $((NOW - START)) -ge "$TIMEOUT" ]; then
      echo "❌ Postgres not ready within ${TIMEOUT}s. Check POSTGRES_URI or DB logs."
      exit 1
    fi
    sleep 1
  done
  echo "✅ Postgres is ready."
fi

# --- (최초 1회) 마이그레이션 실행(선택) ---
if [ -n "${MIGRATE_CMD:-}" ]; then
  echo "🛠  Running migrations..."
  # heredoc/multiline 안전 실행
  sh -lc "$MIGRATE_CMD"
  echo "✅ Migrations completed."
fi

# --- API 기동 ---
echo "🚀 Starting API..."
exec uvicorn estalan.deployment.server:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --no-access-log \
  --timeout-graceful-shutdown 3600 \
  ${RELOAD:-}
