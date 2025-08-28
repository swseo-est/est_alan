#!/bin/bash
# Estsoft Alan API Service Integrated Build and Scaling Script

echo "Estsoft Alan API Service Management Script"
echo

# 기본 replica 수
DEFAULT_REPLICAS=3
if [ -f ".env.dockerfile" ]; then
  while IFS='=' read -r key value; do
    if [ "$key" = "ALAN_API_SERVER_REPLICAS" ]; then
      DEFAULT_REPLICAS=$value
    fi
  done < .env.dockerfile
fi

# 명령어 인자 파싱
COMMAND=$1
REPLICAS=$2

# 인자 없을 경우 기본 build 동작
if [ -z "$COMMAND" ]; then
  COMMAND="build"
  REPLICAS=$DEFAULT_REPLICAS
fi

case "$COMMAND" in
  build)
    [ -z "$REPLICAS" ] && REPLICAS=$DEFAULT_REPLICAS
    echo "Building and starting Estsoft Alan service with $REPLICAS instances..."
    docker compose build
    docker compose up -d --scale langgraph-api=$REPLICAS
    ;;
  scale|up)
    [ -z "$REPLICAS" ] && REPLICAS=$DEFAULT_REPLICAS
    echo "Scaling Estsoft Alan service to $REPLICAS instances..."
    docker compose up -d --scale langgraph-api=$REPLICAS
    ;;
  down)
    [ -z "$REPLICAS" ] && REPLICAS=1
    echo "Scaling down Estsoft Alan service to $REPLICAS instances..."
    docker compose up -d --scale langgraph-api=$REPLICAS
    ;;
  monitor)
    echo "=== Estsoft Alan API Service Monitoring ==="
    echo
    echo "1. Overall service status:"
    docker compose ps
    echo
    echo "2. Estsoft Alan API instance details:"
    docker compose ps langgraph-api
    echo
    echo "3. Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
    echo
    echo "4. Recent logs (by instance):"
    docker compose logs --tail=10 langgraph-api
    echo
    echo "5. Health check:"
    if curl -fs http://localhost/health > /dev/null; then
      echo "✅ Load balancer health check successful"
    else
      echo "❌ Load balancer health check failed"
    fi
    if curl -fs http://localhost/nginx_status > /dev/null; then
      echo "✅ Nginx status page accessible"
    else
      echo "❌ Nginx status page not accessible"
    fi
    echo
    echo "=== Monitoring completed ==="
    ;;
  stop)
    echo "Stopping all Estsoft Alan services..."
    docker compose down
    echo "Estsoft Alan services stopped."
    ;;
  restart)
    echo "Restarting all Estsoft Alan services..."
    docker compose restart
    echo "Estsoft Alan services restarted."
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Run ./build.sh without arguments to auto-build with default replicas."
    exit 1
    ;;
esac

echo
echo "Service status check:"
docker compose ps

echo
echo "Operation completed."
echo "Estsoft Alan load balancer accessible at: http://localhost:80"
echo "Health check: http://localhost/health"
echo "Nginx status: http://localhost/nginx_status"
