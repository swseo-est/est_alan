@echo off
REM storage/compose.yml을 사용하여 도커 컴포즈 빌드 및 컨테이너 실행
docker-compose build --no-cache
docker-compose up -d

docker-compose ps
pause