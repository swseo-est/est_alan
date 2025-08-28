#!/bin/bash

# API 서버 로그 보기 스크립트
# Usage: ./logs.sh [service_name] [options]
# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 도움말 함수
show_help() {
    echo -e "${BLUE}API 서버 로그 보기 스크립트${NC}"
    echo ""
    echo "사용법:"
    echo "  ./logs.sh                    # 모든 서비스의 로그 보기"
    echo "  ./logs.sh [service_name]     # 특정 서비스의 로그 보기"
    echo "  ./logs.sh -f [service_name]  # 실시간 로그 보기 (follow)"
    echo "  ./logs.sh -t [service_name]  # 마지막 N줄 보기"
    echo ""
    echo "서비스 목록:"
    echo "  langgraph-api      # 메인 API 서버"
    echo "  langgraph-redis    # Redis 서버"
    echo "  langgraph-postgres # PostgreSQL 서버"
    echo ""
    echo "옵션:"
    echo "  -f, --follow       # 실시간 로그 보기"
    echo "  -t, --tail N       # 마지막 N줄 보기 (기본값: 100)"
    echo "  -l, --level LEVEL  # 로그 레벨 필터링 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    echo "  -h, --help         # 도움말 보기"
    echo ""
    echo "로그 레벨:"
    echo "  DEBUG     # 디버그 정보 (가장 상세)"
    echo "  INFO      # 일반 정보"
    echo "  WARNING   # 경고"
    echo "  ERROR     # 오류"
    echo "  CRITICAL  # 치명적 오류 (가장 중요)"
    echo ""
    echo "예시:"
    echo "  ./logs.sh -l ERROR langgraph-api     # API 서버의 오류 로그만 보기"
    echo "  ./logs.sh -f -l WARNING              # 모든 서비스의 경고 이상 로그 실시간 보기"
    echo ""
}

# 서비스 목록
SERVICES=("langgraph-api" "langgraph-redis" "langgraph-postgres")

# 로그 레벨 정의
LOG_LEVELS=("DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL")

# 기본값 설정
FOLLOW=false
TAIL_LINES=100
SERVICE_NAME=""
LOG_LEVEL=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -t|--tail)
            TAIL_LINES="$2"
            shift 2
            ;;
        -l|--level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$SERVICE_NAME" ]]; then
                SERVICE_NAME="$1"
            else
                echo -e "${RED}서비스 이름은 하나만 지정할 수 있습니다.${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# 서비스 유효성 검사
if [[ -n "$SERVICE_NAME" ]]; then
    valid_service=false
    for service in "${SERVICES[@]}"; do
        if [[ "$SERVICE_NAME" == "$service" ]]; then
            valid_service=true
            break
        fi
    done
    
    if [[ "$valid_service" == false ]]; then
        echo -e "${RED}유효하지 않은 서비스 이름: $SERVICE_NAME${NC}"
        echo "사용 가능한 서비스: ${SERVICES[*]}"
        exit 1
    fi
fi

# 로그 레벨 유효성 검사
if [[ -n "$LOG_LEVEL" ]]; then
    valid_level=false
    for level in "${LOG_LEVELS[@]}"; do
        if [[ "$LOG_LEVEL" == "$level" ]]; then
            valid_level=true
            break
        fi
    done
    
    if [[ "$valid_level" == false ]]; then
        echo -e "${RED}유효하지 않은 로그 레벨: $LOG_LEVEL${NC}"
        echo "사용 가능한 로그 레벨: ${LOG_LEVELS[*]}"
        exit 1
    fi
fi

# Docker Compose 상태 확인
if ! docker compose ps > /dev/null 2>&1; then
    echo -e "${RED}Docker Compose가 실행되지 않고 있습니다.${NC}"
    echo "먼저 ./build.sh를 실행하여 서비스를 시작하세요."
    exit 1
fi

# 로그 보기 함수
view_logs() {
    local service=$1
    local follow=$2
    local tail_lines=$3
    local log_level=$4
    
    echo -e "${GREEN}=== $service 로그 ===${NC}"
    if [[ -n "$log_level" ]]; then
        echo -e "${YELLOW}로그 레벨 필터: $log_level 이상${NC}"
    fi
    echo ""
    
    # 로그 레벨 필터링을 위한 grep 패턴 생성
    local grep_pattern=""
    if [[ -n "$log_level" ]]; then
        case "$log_level" in
            "DEBUG")
                grep_pattern="(DEBUG|INFO|WARNING|ERROR|CRITICAL)"
                ;;
            "INFO")
                grep_pattern="(INFO|WARNING|ERROR|CRITICAL)"
                ;;
            "WARNING")
                grep_pattern="(WARNING|ERROR|CRITICAL)"
                ;;
            "ERROR")
                grep_pattern="(ERROR|CRITICAL)"
                ;;
            "CRITICAL")
                grep_pattern="CRITICAL"
                ;;
        esac
    fi
    
    if [[ "$follow" == true ]]; then
        echo -e "${YELLOW}실시간 로그를 보여줍니다. 종료하려면 Ctrl+C를 누르세요.${NC}"
        echo ""
        if [[ -n "$grep_pattern" ]]; then
            docker compose logs -f "$service" | grep -E "$grep_pattern"
        else
            docker compose logs -f "$service"
        fi
    else
        if [[ -n "$grep_pattern" ]]; then
            docker compose logs --tail="$tail_lines" "$service" | grep -E "$grep_pattern"
        else
            docker compose logs --tail="$tail_lines" "$service"
        fi
    fi
}

# 메인 로직
if [[ -z "$SERVICE_NAME" ]]; then
    # 모든 서비스 로그 보기
    echo -e "${BLUE}모든 서비스의 로그를 보여줍니다.${NC}"
    echo ""
    
    for service in "${SERVICES[@]}"; do
        view_logs "$service" "$FOLLOW" "$TAIL_LINES" "$LOG_LEVEL"
        echo ""
        echo -e "${BLUE}----------------------------------------${NC}"
        echo ""
    done
else
    # 특정 서비스 로그 보기
    view_logs "$SERVICE_NAME" "$FOLLOW" "$TAIL_LINES" "$LOG_LEVEL"
fi
