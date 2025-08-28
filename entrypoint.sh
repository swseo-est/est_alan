#!/bin/sh
set -e

if [ "$RUN_MIGRATIONS" = "1" ]; then
  echo "[Entrypoint] Running DB migrations..."
  python ./estalan/deployment/migrate.py
  echo "[Entrypoint] DB migrations finished. Exiting."
  exit 0
fi

echo "[Entrypoint] Starting API server..."
export LANGSERVE_GRAPHS="$(cat /graph.json)"
exec uvicorn estalan.deployment.server:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --no-access-log \
  --timeout-graceful-shutdown 3600 \
  $RELOAD
