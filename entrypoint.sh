#!/bin/sh
export LANGSERVE_GRAPHS="$(cat /graph.json)"
exec uvicorn estalan.deployment.server:app --host 0.0.0.0 --port $PORT --no-access-log --timeout-graceful-shutdown 3600 $RELOAD
