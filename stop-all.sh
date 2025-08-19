#!/bin/bash
# 현재 폴더의 docker-compose.yml 기준으로 모든 컨테이너 중지
set -e

echo "Stopping all containers for project: $(basename "$PWD")"
docker compose down
