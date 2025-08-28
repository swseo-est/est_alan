@echo off
REM Estsoft Alan API Service Integrated Build and Scaling Script

echo Estsoft Alan API Service Management Script
echo.

REM Read default replica count from environment file
set DEFAULT_REPLICAS=3
if exist .env.dockerfile (
    for /f "tokens=1* delims==" %%a in (.env.dockerfile) do (
        if "%%a"=="ALAN_API_SERVER_REPLICAS" set DEFAULT_REPLICAS=%%b
    )
)

REM Command parsing
set COMMAND=%1
set REPLICAS=%2

REM 인자가 없을 경우 기본 build 동작
if "%COMMAND%"=="" (
    set COMMAND=build
    set REPLICAS=%DEFAULT_REPLICAS%
)

REM 분기 처리
if /i "%COMMAND%"=="build"    goto :BUILD
if /i "%COMMAND%"=="scale"    goto :SCALE
if /i "%COMMAND%"=="up"       goto :UP
if /i "%COMMAND%"=="down"     goto :DOWN
if /i "%COMMAND%"=="monitor"  goto :MONITOR
if /i "%COMMAND%"=="stop"     goto :STOP
if /i "%COMMAND%"=="restart"  goto :RESTART

echo Unknown command: %COMMAND%
echo Run build.bat without arguments to auto-build with default replicas.
pause
goto :EOF

:BUILD
if "%REPLICAS%"=="" set REPLICAS=%DEFAULT_REPLICAS%
echo Building and starting Estsoft Alan service with %REPLICAS% instances...
docker compose build
docker compose up -d --scale langgraph-api=%REPLICAS%
goto :END

:SCALE
if "%REPLICAS%"=="" set REPLICAS=%DEFAULT_REPLICAS%
echo Scaling Estsoft Alan service to %REPLICAS% instances...
docker compose up -d --scale langgraph-api=%REPLICAS%
goto :END

:UP
if "%REPLICAS%"=="" set REPLICAS=%DEFAULT_REPLICAS%
echo Scaling up Estsoft Alan service to %REPLICAS% instances...
docker compose up -d --scale langgraph-api=%REPLICAS%
goto :END

:DOWN
if "%REPLICAS%"=="" set REPLICAS=1
echo Scaling down Estsoft Alan service to %REPLICAS% instances...
docker compose up -d --scale langgraph-api=%REPLICAS%
goto :END

:MONITOR
echo === Estsoft Alan API Service Monitoring ===
echo.
echo 1. Overall service status:
docker compose ps
echo.
echo 2. Estsoft Alan API instance details:
docker compose ps langgraph-api
echo.
echo 3. Resource usage:
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
echo.
echo 4. Recent logs (by instance):
docker compose logs --tail=10 langgraph-api
echo.
echo 5. Health check:
curl -f http://localhost/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Load balancer health check failed
) else (
    echo ✅ Load balancer health check successful
)
curl -f http://localhost/nginx_status >nul 2>&1
if errorlevel 1 (
    echo ❌ Nginx status page not accessible
) else (
    echo ✅ Nginx status page accessible
)
echo.
echo === Monitoring completed ===
pause
goto :EOF

:STOP
echo Stopping all Estsoft Alan services...
docker compose down
echo Estsoft Alan services stopped.
pause
goto :EOF

:RESTART
echo Restarting all Estsoft Alan services...
docker compose restart
echo Estsoft Alan services restarted.
pause
goto :EOF

:END
echo.
echo Service status check:
docker compose ps

echo.
echo Operation completed.
echo Estsoft Alan load balancer accessible at: http://localhost:80
echo Health check: http://localhost/health
echo Nginx status: http://localhost/nginx_status

pause
goto :EOF
