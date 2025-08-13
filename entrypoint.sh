#!/usr/bin/env sh
set -e

# 기존 동작 유지
export LANGSERVE_GRAPHS="$(cat /graph.json)"

# --- DB Ready 대기 ---
if [ -n "${POSTGRES_URI:-}" ]; then
  echo "⏳ Waiting for Postgres to be ready..."
  # pg_isready는 libpq URI를 -d 로 받을 수 있습니다.
  until pg_isready -d "$POSTGRES_URI" >/dev/null 2>&1; do
    sleep 1
  done
fi

# --- 선택: 마이그레이션 실행 ---
if [ -n "${MIGRATE_CMD:-}" ]; then
  echo "🛠  Running migrations: $MIGRATE_CMD"
  sh -lc "$MIGRATE_CMD"
fi

echo "🚀 Starting API..."
exec uvicorn estalan.deployment.server:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --no-access-log \
  --timeout-graceful-shutdown 3600 \
  ${RELOAD:-}
