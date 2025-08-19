import contextlib
import json
import logging
import os
import pathlib
import threading
import typing
from collections.abc import Mapping, Sequence
from typing import Literal

if typing.TYPE_CHECKING:
    from estalan.deployment.config import HttpConfig, StoreConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@contextlib.contextmanager
def patch_environment(**kwargs):
    """Temporarily patch environment variables.

    Args:
        **kwargs: Key-value pairs of environment variables to set.

    Yields:
        None
    """
    original = {}
    try:
        for key, value in kwargs.items():
            if value is None:
                original[key] = os.environ.pop(key, None)
                continue
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value



def run_server(
    # 인증 관련 설정: 서버 바인딩 호스트
    # "127.0.0.1" (localhost): 로컬 접근만 허용 (기본값, 가장 안전)
    # "0.0.0.0": 모든 네트워크 인터페이스에서 접근 허용 (프로덕션 환경)
    # 특정 IP: 해당 IP에서만 접근 허용
    host: str = "127.0.0.1",
    port: int = 2024,
    reload: bool = False,
    graphs: dict | None = None,
    n_jobs_per_worker: int | None = None,
    env_file: str | None = None,
    open_browser: bool = False,
    tunnel: bool = False,
    debug_port: int | None = None,
    wait_for_client: bool = False,
    env: str | pathlib.Path | Mapping[str, str] | None = None,
    reload_includes: Sequence[str] | None = None,
    reload_excludes: Sequence[str] | None = None,
    store: typing.Optional["StoreConfig"] = None,
    http: typing.Optional["HttpConfig"] = None,
    ui: dict | None = None,
    ui_config: dict | None = None,
    disable_persistence: bool = False,
    allow_blocking: bool = False,
    runtime_edition: Literal["inmem", "community", "postgres"] = "inmem",
    server_level: str = "WARNING",
    **kwargs: typing.Any,
):
    """Run the LangGraph API server."""

    import inspect
    import time

    import uvicorn

    start_time = time.time()

    env_vars = env if isinstance(env, Mapping) else None
    # 인증 관련 설정: API 경로 접두사 설정
    # 마이크로서비스 환경에서 API를 특정 경로에 마운트할 때 사용 (예: /api/v1, /auth/api 등)
    # 인증 미들웨어나 프록시와 함께 사용하여 특정 경로에 대한 인증을 적용할 수 있음
    mount_prefix = None
    if http is not None and http.get("mount_prefix") is not None:
        mount_prefix = http.get("mount_prefix")
    if os.environ.get("LANGGRAPH_MOUNT_PREFIX"):
        mount_prefix = os.environ.get("LANGGRAPH_MOUNT_PREFIX")
    if isinstance(env, str | pathlib.Path):
        try:
            from dotenv.main import DotEnv

            env_vars = DotEnv(dotenv_path=env).dict() or {}
            logger.debug(f"Loaded environment variables from {env}: {sorted(env_vars)}")

        except ImportError:
            logger.warning(
                "python_dotenv is not installed. Environment variables will not be available."
            )

    if debug_port is not None:
        try:
            import debugpy
        except ImportError:
            logger.warning("debugpy is not installed. Debugging will not be available.")
            logger.info("To enable debugging, install debugpy: pip install debugpy")
            return
        debugpy.listen((host, debug_port))
        logger.info(
            f"🐛 Debugger listening on port {debug_port}. Waiting for client to attach..."
        )
        logger.info("To attach the debugger:")
        logger.info("1. Open your python debugger client (e.g., Visual Studio Code).")
        logger.info(
            "2. Use the 'Remote Attach' configuration with the following settings:"
        )
        logger.info("   - Host: 0.0.0.0")
        logger.info(f"   - Port: {debug_port}")
        logger.info("3. Start the debugger to connect to the server.")
        if wait_for_client:
            debugpy.wait_for_client()
            logger.info("Debugger attached. Starting server...")

    # Determine local or tunneled URL
    upstream_url = f"http://{host}:{port}"
    if mount_prefix:
        upstream_url += mount_prefix
    if tunnel:
        # 인증 관련 설정: Cloudflare Tunnel을 통한 보안 연결
        # 로컬 서버를 인터넷에 안전하게 노출시키는 방법
        # Cloudflare의 인증 및 보안 기능을 활용하여 HTTPS 연결 제공
        logger.info("Starting Cloudflare Tunnel...")
        from concurrent.futures import TimeoutError as FutureTimeoutError

        from langgraph_api.tunneling.cloudflare import start_tunnel

        tunnel_obj = start_tunnel(port)
        try:
            public_url = tunnel_obj.url.result(timeout=30)
        except FutureTimeoutError:
            logger.warning(
                "Timed out waiting for Cloudflare Tunnel URL; using local URL %s",
                upstream_url,
            )
            public_url = upstream_url
        except Exception as e:
            tunnel_obj.process.kill()
            raise RuntimeError("Failed to start Cloudflare Tunnel") from e
        local_url = public_url
        if mount_prefix:
            local_url += mount_prefix
    else:
        local_url = upstream_url
    to_patch = dict(
        MIGRATIONS_PATH="__inmem",
        DATABASE_URI=":memory:",
        REDIS_URI="fake",
        N_JOBS_PER_WORKER=str(n_jobs_per_worker if n_jobs_per_worker else 1),
        LANGGRAPH_STORE=json.dumps(store) if store else None,
        LANGSERVE_GRAPHS=json.dumps(graphs) if graphs else None,


        LANGGRAPH_HTTP=json.dumps(http) if http else None,
        LANGGRAPH_UI=json.dumps(ui) if ui else None,
        LANGGRAPH_UI_CONFIG=json.dumps(ui_config) if ui_config else None,
        LANGGRAPH_UI_BUNDLER="true",
        LANGGRAPH_API_URL=local_url,
        LANGGRAPH_DISABLE_FILE_PERSISTENCE=str(disable_persistence).lower(),
        LANGGRAPH_RUNTIME_EDITION=runtime_edition,
        # If true, we will not raise on blocking IO calls (via blockbuster)
        LANGGRAPH_ALLOW_BLOCKING=str(allow_blocking).lower(),
        # 인증 관련 설정: 프라이빗 네트워크 접근 허용
        # Chrome의 Private Network Access 정책에 따라 프라이빗 네트워크(192.168.x.x, 10.x.x.x 등)에서의 접근을 허용
        # See https://developer.chrome.com/blog/private-network-access-update-2024-03
        ALLOW_PRIVATE_NETWORK="true",
    )
    if env_vars is not None:
        # LangSmith 설정 병합: 로드된 환경변수를 서버 설정에 병합
        # LANGCHAIN_* 환경변수들이 서버 실행 중에 사용 가능하도록 설정
        # Don't overwrite.
        for k, v in env_vars.items():
            if k in to_patch:
                logger.debug(f"Skipping loaded env var {k}={v}")
                continue
            to_patch[k] = v
    with patch_environment(
        **to_patch,
    ):
        def _open_browser():
            # 브라우저 열기 함수: 서버 시작 후 API 문서를 자동으로 브라우저에서 열기
            import time
            import urllib.request
            import webbrowser

            thread_logger = logging.getLogger("browser_opener")
            if not thread_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                thread_logger.addHandler(handler)

            while True:
                try:
                    with urllib.request.urlopen(f"{local_url}/ok") as response:
                        if response.status == 200:
                            thread_logger.info(
                                f"Server started in {time.time() - start_time:.2f}s"
                            )
                            thread_logger.info(
                                "📚 Opening API docs in your browser..."
                            )
                            # API 문서 URL 로그 출력 및 브라우저에서 열기
                            api_docs_url = f"{local_url}/docs"
                            thread_logger.info("URL: " + api_docs_url)
                            webbrowser.open(api_docs_url)
                            return
                except urllib.error.URLError:
                    pass
                time.sleep(0.1)

        # 웰컴 메시지: 서버 시작 시 표시되는 정보
        # API URL과 API 문서 URL을 포함한 서버 정보 출력
        welcome = f"""

        Welcome to

███████ ███████ ████████ ███████  ██████  ███████ ████████      █████  ██       █████  ███    ██ 
██      ██         ██    ██      ██    ██ ██         ██        ██   ██ ██      ██   ██ ████   ██ 
█████   ███████    ██    ███████ ██    ██ █████      ██        ███████ ██      ███████ ██ ██  ██ 
██           ██    ██         ██ ██    ██ ██         ██        ██   ██ ██      ██   ██ ██  ██ ██ 
███████ ███████    ██    ███████  ██████  ██         ██        ██   ██ ███████ ██   ██ ██   ████ 
                                                                                                 
                                                         
- 🚀 API: \033[36m{local_url}\033[0m
- 📚 API Docs: \033[36m{local_url}/docs\033[0m

This in-memory server is designed for development and testing.
"""
        logger.info(welcome)
        # 브라우저 자동 열기: open_browser가 True인 경우 백그라운드 스레드로 API 문서 열기
        # 서버가 완전히 시작된 후 자동으로 브라우저에서 API 문서를 열어줌
        if open_browser:
            threading.Thread(target=_open_browser, daemon=True).start()
        supported_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(uvicorn.run).parameters
        }
        server_level = server_level.upper()
        uvicorn.run(
            "estalan.deployment.server:app",
            host=host,
            port=port,
            reload=reload,
            env_file=env_file,
            access_log=False,
            reload_includes=reload_includes,
            reload_excludes=reload_excludes,
            log_config={
                "version": 1,
                "incremental": False,
                "disable_existing_loggers": False,
                "formatters": {
                    "simple": {
                        "class": "langgraph_api.logging.Formatter",
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "simple",
                        "stream": "ext://sys.stdout",
                    }
                },
                "loggers": {
                    "uvicorn": {"level": server_level},
                    "uvicorn.error": {"level": server_level},
                    "estalan.deployment.server": {"level": server_level},
                },
                "root": {"handlers": ["console"]},
            },
            **supported_kwargs,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI entrypoint for running the LangGraph API server."
    )
    parser.add_argument(
        # 인증 관련 설정: 서버 바인딩 호스트
        # "127.0.0.1": 로컬 접근만 허용 (기본값, 개발 환경에 적합)
        # "0.0.0.0": 모든 네트워크에서 접근 허용 (프로덕션 환경)
        "--host", default="127.0.0.1", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=2024, help="Port to bind the server to"
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument(
        # LangSmith 설정: 설정 파일 경로
        # langgraph.json 파일에 LangSmith 관련 설정을 포함할 수 있음
        # "env" 섹션에 LANGCHAIN_* 환경변수들을 정의 가능
        "--config", default="langgraph.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--n-jobs-per-worker",
        type=int,
        help="Number of jobs per worker. Default is None (meaning 10)",
    )
    parser.add_argument(
        # 브라우저 설정: API 문서 자동 열기 비활성화
        # --no-browser 옵션을 사용하면 서버 시작 후 API 문서가 자동으로 브라우저에서 열리지 않음
        "--no-browser", action="store_true", help="Disable automatic browser opening"
    )
    parser.add_argument(
        "--debug-port", type=int, help="Port for debugger to listen on (default: none)"
    )
    parser.add_argument(
        "--wait-for-client",
        action="store_true",
        help="Whether to break and wait for a debugger to attach",
    )
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Expose the server via Cloudflare Tunnel",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_data = json.load(f)

    graphs = config_data.get("graphs", {})
    ui = config_data.get("ui")
    ui_config = config_data.get("ui_config")
    # LangSmith 설정: 설정 파일에서 환경변수 로드
    # config_data.get("env")에서 LANGCHAIN_* 환경변수들을 가져와서 서버에 전달
    run_server(
        args.host,
        args.port,
        not args.no_reload,
        graphs,
        n_jobs_per_worker=args.n_jobs_per_worker,
        # 브라우저 설정: CLI 인자에 따라 API 문서 자동 열기 여부 결정
        # --no-browser 옵션이 있으면 False, 없으면 True로 설정
        open_browser=not args.no_browser,
        tunnel=args.tunnel,
        debug_port=args.debug_port,
        wait_for_client=args.wait_for_client,
        env=config_data.get("env", None),
        ui=ui,
        ui_config=ui_config,
    )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    
    sys.argv = [
        "estalan.deployment.cli",  # 스크립트 이름
        "--config", "./test/simple_chat_model/langgraph.json",
    ]

    main()
