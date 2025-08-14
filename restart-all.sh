#!/bin/bash
# 현재 폴더의 docker-compose.yml 기준으로 컨테이너 재시작
set -e

echo "Restarting all containers for project: $(basename "$PWD")"
docker compose down
docker compose up -d --build
