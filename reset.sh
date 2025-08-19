#!/usr/bin/env bash
set -euo pipefail

echo "?? 모든 Docker 리소스를 정리합니다..."

# 1) 중지/삭제
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm -f $(docker ps -aq) 2>/dev/null || true
docker volume rm $(docker volume ls -q) 2>/dev/null || true
docker network rm $(docker network ls -q | grep -vE '^(bridge|host|none)$') 2>/dev/null || true
docker rmi -f $(docker images -q) 2>/dev/null || true
docker builder prune -a -f || true
docker system prune -a --volumes -f || true

# 2) 서비스 중지 후 데이터 디렉터리 제거
echo "?? /var/lib/docker 및 /var/lib/containerd 삭제"
sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
sudo rm -rf /var/lib/docker /var/lib/containerd
rm -rf ~/.docker ~/.cache/buildx ~/.local/share/buildx 2>/dev/null || true

# 3) 서비스 재시작(도커 계속 쓸 거라면)
sudo systemctl start containerd 2>/dev/null || true
sudo systemctl start docker 2>/dev/null || true

echo "? Docker 리셋 완료"
