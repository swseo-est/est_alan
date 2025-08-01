# MONKEY PATCH: Patch Starlette to fix an error in the library
# ruff: noqa: E402
import langgraph_api.patch  # noqa: F401,I001
import sys
import os

# WARNING: Keep the import above before other code runs as it
# patches an error in the Starlette library.
import logging
import typing

if not (
    (disable_truststore := os.getenv("DISABLE_TRUSTSTORE"))
    and disable_truststore.lower() == "true"
):
    import truststore  # noqa: F401

    truststore.inject_into_ssl()  # noqa: F401

from contextlib import asynccontextmanager

import jsonschema_rs
import structlog
from langgraph.errors import EmptyInputError, InvalidUpdateError
from langgraph_sdk.client import configure_loopback_transports
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

import estalan.deployment.config as config
from langgraph_api.api import meta_routes, routes, user_router
from langgraph_api.api.openapi import set_custom_spec
from langgraph_api.errors import (
    overloaded_error_handler,
    validation_error_handler,
    value_error_handler,
)
from langgraph_api.js.base import is_js_path
from langgraph_api.middleware.http_logger import AccessLoggerMiddleware
from langgraph_api.middleware.private_network import PrivateNetworkMiddleware
from langgraph_api.middleware.request_id import RequestIdMiddleware
from langgraph_api.utils import SchemaGenerator
from langgraph_runtime.lifespan import lifespan
from langgraph_runtime.retry import OVERLOADED_EXCEPTIONS

logging.captureWarnings(True)
logger = logging.getLogger("langgraph_license.validation")
logger.disabled = True

logger = structlog.stdlib.get_logger(__name__)
welcome = f"""

███████ ███████ ████████ ███████  ██████  ███████ ████████      █████  ██       █████  ███    ██ 
██      ██         ██    ██      ██    ██ ██         ██        ██   ██ ██      ██   ██ ████   ██ 
█████   ███████    ██    ███████ ██    ██ █████      ██        ███████ ██      ███████ ██ ██  ██ 
██           ██    ██         ██ ██    ██ ██         ██        ██   ██ ██      ██   ██ ██  ██ ██ 
███████ ███████    ██    ███████  ██████  ██         ██        ██   ██ ███████ ██   ██ ██   ████ 

"""

logger.info(welcome)
middleware = []

if config.ALLOW_PRIVATE_NETWORK:
    middleware.append(Middleware(PrivateNetworkMiddleware))

if (
    config.HTTP_CONFIG
    and (app := config.HTTP_CONFIG.get("app"))
    and is_js_path(app.split(":")[0])
):
    from langgraph_api.js.remote import JSCustomHTTPProxyMiddleware

    middleware.append(Middleware(JSCustomHTTPProxyMiddleware))

middleware.extend(
    [
        (
            Middleware(
                CORSMiddleware,
                allow_origins=config.CORS_ALLOW_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["x-pagination-total"],
            )
            if config.CORS_CONFIG is None
            else Middleware(
                CORSMiddleware,
                **config.CORS_CONFIG,
            )
        ),
        Middleware(AccessLoggerMiddleware, logger=logger),
        Middleware(RequestIdMiddleware, mount_prefix=config.MOUNT_PREFIX),
    ]
)
exception_handlers = {
    ValueError: value_error_handler,
    InvalidUpdateError: value_error_handler,
    EmptyInputError: value_error_handler,
    jsonschema_rs.ValidationError: validation_error_handler,
} | {exc: overloaded_error_handler for exc in OVERLOADED_EXCEPTIONS}


def update_openapi_spec(app):
    """
    OpenAPI 스펙을 업데이트하는 함수
    
    FastAPI 앱인 경우 기존 OpenAPI 스펙을 사용하고,
    그렇지 않은 경우 새로운 스키마를 생성합니다.
    
    Args:
        app: FastAPI 또는 Starlette 애플리케이션 인스턴스
    """
    spec = None
    if "fastapi" in sys.modules:
        # It's maybe a fastapi app
        from fastapi import FastAPI

        if isinstance(user_router, FastAPI):
            spec = app.openapi()

    if spec is None:
        # How do we add
        schemas = SchemaGenerator(
            {
                "openapi": "3.1.0",
                "info": {"title": "LangGraph Platform", "version": "0.1.0"},
            }
        )
        spec = schemas.get_schema(routes=app.routes)

    if spec:
        set_custom_spec(spec)


if user_router:
    # Merge routes
    app = user_router

    meta_route_paths = [route.path for route in meta_routes]
    custom_route_paths = [
        route.path
        for route in user_router.router.routes
        if route.path not in meta_route_paths
    ]
    logger.info(f"Custom route paths: {custom_route_paths}")

    update_openapi_spec(app)
    for route in routes:
        if route.path in ("/docs", "/openapi.json"):
            # Our handlers for these are inclusive of the custom routes and default API ones
            # Don't let these be shadowed
            app.router.routes.insert(0, route)
        else:
            # Everything else could be shadowed.
            app.router.routes.append(route)

    # Merge lifespans
    original_lifespan = app.router.lifespan_context
    if app.router.on_startup or app.router.on_shutdown:
        raise ValueError(
            f"Cannot merge lifespans with on_startup or on_shutdown: {app.router.on_startup} {app.router.on_shutdown}"
        )

    @asynccontextmanager
    async def combined_lifespan(app):
        """
        사용자 정의 lifespan과 기본 lifespan을 결합하는 비동기 컨텍스트 매니저
        
        두 개의 lifespan을 순차적으로 실행하여 애플리케이션의 시작과 종료를 관리합니다.
        
        Args:
            app: 애플리케이션 인스턴스
            
        Yields:
            None: 컨텍스트 매니저가 활성화된 상태
        """
        async with lifespan(app):
            if original_lifespan:
                async with original_lifespan(app):
                    yield
            else:
                yield

    app.router.lifespan_context = combined_lifespan

    # Merge middleware
    app.user_middleware = (app.user_middleware or []) + middleware
    # Merge exception handlers
    for k, v in exception_handlers.items():
        if k not in app.exception_handlers:
            app.exception_handlers[k] = v
        else:
            logger.debug(f"Overriding exception handler for {k}")
    # If the user creates a loopback client with `get_client() (no url)
    # this will update the http transport to connect to the right app
    configure_loopback_transports(app)

else:
    # It's a regular starlette app
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=middleware,
        exception_handlers=exception_handlers,
    )

if config.MOUNT_PREFIX:
    prefix = config.MOUNT_PREFIX
    if not prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError(
            f"Invalid mount_prefix '{prefix}': Must start with '/' and must not end with '/'. "
            f"Valid examples: '/my-api', '/v1', '/api/v1'.\nInvalid examples: 'api/', '/api/'"
        )
    logger.info(f"Mounting routes at prefix: {prefix}")
    plen = len(prefix)
    rplen = len(prefix.encode("utf-8"))

    class ASGIBypassMiddleware:
        """
        ASGI 미들웨어 클래스
        
        특정 경로 패턴에 대해 ASGI 요청을 우회하고 올바른 경로로 리다이렉트합니다.
        주로 인증이 필요 없는 경로('/noauth')에 대한 처리를 담당합니다.
        """
        def __init__(self, app: typing.Any, **kwargs):
            """
            ASGIBypassMiddleware 초기화
            
            Args:
                app: ASGI 애플리케이션 인스턴스
                **kwargs: 추가 키워드 인자
            """
            self.app = app

        async def __call__(
            self, scope: Scope, receive: Receive, send: Send
        ) -> typing.Any:
            """
            ASGI 요청을 처리하는 메서드
            
            '/noauth' 경로에 대한 요청을 감지하고 적절한 경로로 변환합니다.
            
            Args:
                scope: ASGI 스코프 정보
                receive: 요청 수신 함수
                send: 응답 전송 함수
                
            Returns:
                ASGI 애플리케이션의 응답
            """
            if (root_path := scope.get("root_path")) and root_path == "/noauth":
                # The SDK initialized with None is trying to connect via
                # ASGITransport. Ensure that it has the correct subpath prefixes
                # so the regular router can handle it.
                scope["path"] = f"/noauth{prefix}{scope['path']}"
                scope["raw_path"] = scope["path"].encode("utf-8")

            return await self.app(scope, receive, send)

    app = Starlette(
        routes=[Mount(prefix, app=app)],
        lifespan=app.router.lifespan_context,
        middleware=[Middleware(ASGIBypassMiddleware)] + app.user_middleware,
        exception_handlers=app.exception_handlers,
    )
