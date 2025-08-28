"""
Estalan 로깅 시스템

사용법:
    from estalan.logging import get_logger, EstalanLogger
    
    # 로거 생성
    logger = get_logger(__name__)
    logger.info("메시지")
    
    # 또는 직접 클래스 사용
    logger = EstalanLogger(__name__)
    logger.info("메시지")
"""

from .base import (
    EstalanLogger,
    ColoredFormatter,
    JSONFormatter,
    get_logger,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
)

__all__ = [
    "EstalanLogger",
    "ColoredFormatter", 
    "JSONFormatter",
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
]
