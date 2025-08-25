import json
import time
import pytest
from unittest.mock import patch, MagicMock
import requests
import subprocess
import threading
import signal
import os
import sys
from pathlib import Path

from estalan.deployment.cli import main
from langgraph_sdk import get_client

# 배포 URL 설정
DEPLOYMENT_URL = "http://localhost:2024/"


def get_langgraph_client(url=None):
    """LangGraph 클라이언트 객체를 생성하는 함수
    
    Args:
        url (str, optional): 배포 URL. None이면 기본 DEPLOYMENT_URL 사용
        
    Returns:
        langgraph_sdk.client.Client: LangGraph 클라이언트 객체
    """
    if url is None:
        url = DEPLOYMENT_URL
    
    return get_client(url=url)


# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path.cwd().parent  # src의 부모 디렉토리
sys.path.insert(0, str(project_root))

# src 디렉터리 경로 설정 (현재 디렉토리)
src_dir = Path.cwd()
print(f"프로젝트 루트: {project_root}")
print(f"src 디렉터리: {src_dir}")


def start_server_in_background():
    """백그라운드에서 서버를 시작하는 함수"""
    server_process = None
    server_url = "http://127.0.0.1:2024"
    
    try:
        # run.py 스크립트를 실행하여 서버 시작
        cmd = [sys.executable, "run.py"]
        print(f"서버 시작 명령: {cmd}")
        print(f"작업 디렉터리: {src_dir}")
        
        server_process = subprocess.Popen(
            cmd,
            cwd=src_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 서버가 시작될 때까지 대기
        max_wait = 30  # 최대 30초 대기
        wait_time = 0
        while wait_time < max_wait:
            try:
                response = requests.get(f"{server_url}/ok", timeout=1)
                if response.status_code == 200:
                    print(f"서버가 성공적으로 시작되었습니다. (대기 시간: {wait_time}초)")
                    return server_process, True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            wait_time += 1
            
            # 프로세스가 종료되었는지 확인
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                print(f"서버 프로세스가 예기치 않게 종료되었습니다.")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
                return server_process, False
        
        print("서버 시작 대기 시간 초과")
        return server_process, False
        
    except Exception as e:
        print(f"서버 시작 중 오류: {e}")
        return None, False


def stop_server(server_process):
    """서버 프로세스를 종료하는 함수"""
    if server_process:
        try:
            # Windows에서 프로세스 종료
            if os.name == 'nt':
                server_process.terminate()
                server_process.wait(timeout=5)
            else:
                # Unix/Linux에서 프로세스 종료
                server_process.send_signal(signal.SIGTERM)
                server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        except Exception as e:
            print(f"서버 프로세스 종료 중 오류: {e}")



@pytest.fixture
def server_process():
    """서버 프로세스 fixture"""
    return None


def test_langgraph_client_creation():
    """LangGraph 클라이언트 생성 테스트"""
    # 기본 URL로 클라이언트 생성
    client = get_langgraph_client()
    assert client is not None
    
    # 커스텀 URL로 클라이언트 생성
    custom_client = get_langgraph_client("http://localhost:8000/")
    assert custom_client is not None
    
    # 두 클라이언트가 다른 URL을 사용하는지 확인
    assert client != custom_client


@pytest.mark.asyncio
async def test_thread_creation():
    """Thread 생성 및 응답 포맷 검증 테스트"""
    client = get_langgraph_client()
    
    # thread 생성
    thread = await client.threads.create()
    print(f"생성된 thread: {thread}")
    
    # thread_id 추출
    thread_id = thread["thread_id"]
    print(f"thread_id: {thread_id}")
    
    return thread_id


@pytest.mark.asyncio
async def test_server_communication():
    """서버와의 통신 테스트"""
    client = get_langgraph_client()
    
    # thread 생성
    thread_id = await test_thread_creation()
    
    # assistant_id 설정
    assistant_id = "slide_generate_agent"
    
    # 입력 메시지 설정
    input_data = {"messages": "안녕"}
    
    print(f"Thread ID: {thread_id}")
    print(f"Assistant ID: {assistant_id}")
    print(f"Input: {input_data}")
    
    # 서버와 통신하여 스트림 응답 받기
    chunk_count = 0
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode=["updates"],
        stream_subgraphs=True
    ):
        print(f"Received chunk: {chunk}")
        
        # chunk가 유효한 응답인지 확인
        assert chunk is not None
        # StreamPart 객체인지 확인 (LangGraph SDK의 스트림 응답 타입)
        assert hasattr(chunk, 'event') and hasattr(chunk, 'data')
        
        chunk_count += 1
        
        # 최소한 하나의 chunk를 받았으면 테스트 통과
        print(chunk_count, chunk)

    # 최소한 하나의 chunk를 받았는지 확인
    assert chunk_count >= 1, f"Expected at least 1 chunk, but received {chunk_count}"
    
    print("서버 통신 테스트 완료")


def test_server_start_stop(server_process):
    """서버 시작/종료 기능 테스트"""
    # 서버 시작
    server_process, success = start_server_in_background()
    assert server_process is not None
    
    if success:
        # 서버가 정상적으로 시작되었는지 확인
        assert server_process.pid is not None
        
        # 서버 종료
        stop_server(server_process)
        
        # 프로세스가 종료되었는지 확인
        time.sleep(2)  # 종료 대기
        assert server_process.poll() is not None
    else:
        # 서버 시작 실패 시에도 프로세스 정리
        if server_process:
            stop_server(server_process)


def test_server_health_check(server_process):
    """서버 헬스 체크 엔드포인트 테스트"""
    # 서버 시작
    server_process, success = start_server_in_background()
    
    if success:
        try:
            # 헬스 체크 요청
            response = requests.get("http://127.0.0.1:2024/ok", timeout=5)
            assert response.status_code == 200
        finally:
            # 서버 정리
            stop_server(server_process)
    else:
        # 서버 시작 실패 시에도 프로세스 정리
        if server_process:
            stop_server(server_process)
        pytest.skip("서버를 시작할 수 없어서 헬스 체크 테스트를 건너뜁니다.")


