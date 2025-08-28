"""
Estalan 라이브러리를 위한 고급 로깅 시스템
"""

import inspect
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# 공통 포맷 상수 정의
DEFAULT_LOG_FORMAT = '%(levelname)s - [PID:%(process)d:TID:%(thread)d] - %(asctime)s - %(name)s - [%(filename)s:%(lineno)d:%(funcName)s()] - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

try:
    import colorama
    from colorama import Fore, Style
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """콘솔 출력을 위한 컬러 포맷터"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    
    def __init__(self, use_colors: bool = True):
        # 공통 포맷 사용
        super().__init__(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        self.use_colors = use_colors and COLORAMA_AVAILABLE
        
        if self.use_colors:
            colorama.init(autoreset=True)
    
    def format(self, record: logging.LogRecord) -> str:
        # 기본 포맷 적용
        formatted = super().format(record)
        
        if self.use_colors and record.levelname in self.COLORS:
            # 로그 레벨에 색상 적용
            formatted = f"{self.COLORS[record.levelname]}{formatted}{Style.RESET_ALL}"
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON 형태의 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        }
        
        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 추가 필드가 있는 경우 포함
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class EstalanLogger:
    """Estalan 라이브러리를 위한 고급 로거 클래스"""
    
    def __init__(
        self,
        name: str,
        level: Union[str, int] = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        use_colors: bool = True,
        use_json: bool = False,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Args:
            name: 로거 이름
            level: 로그 레벨
            log_file: 로그 파일 경로
            max_bytes: 로그 파일 최대 크기
            backup_count: 백업 파일 개수
            use_colors: 콘솔 색상 사용 여부
            use_json: JSON 포맷 사용 여부
            enable_console: 콘솔 출력 활성화
            enable_file: 파일 출력 활성화
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_level(level))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포맷터 설정 - 공통 포맷 사용
        if use_json:
            self.console_formatter = JSONFormatter()
            self.file_formatter = JSONFormatter()
        else:
            self.console_formatter = ColoredFormatter(use_colors)
            self.file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        
        # 콘솔 핸들러
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        if enable_file and log_file:
            self._setup_file_handler(log_file, max_bytes, backup_count)
        
        # propagate 설정 (상위 로거로 전파 방지)
        self.logger.propagate = False
    
    def _get_level(self, level: Union[str, int]) -> int:
        """로그 레벨을 정수로 변환"""
        if isinstance(level, int):
            return level
        
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        
        return level_map.get(level.upper(), logging.INFO)
    
    def _setup_file_handler(
        self, 
        log_file: str, 
        max_bytes: int, 
        backup_count: int
    ):
        """파일 핸들러 설정"""
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # RotatingFileHandler 사용
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def _get_caller_info(self):
        """호출 스택에서 실제 사용자 코드 위치 찾기"""
        stack = inspect.stack()
        caller_frame = None
        
        # base.py가 아닌 실제 사용자 코드 찾기
        for frame_info in stack:
            if 'base.py' not in frame_info.filename:
                caller_frame = frame_info
                break
        
        if caller_frame:
            return caller_frame.filename, caller_frame.lineno, caller_frame.function
        else:
            return "unknown", 0, "unknown"

    def _log_with_extra(self, level: int, message: str, **kwargs):
        """추가 필드와 함께 로그 기록"""
        # 항상 호출 스택 추적
        filename, lineno, funcname = self._get_caller_info()
        
        if kwargs:
            # extra_fields를 record에 추가
            record = self.logger.makeRecord(
                self.name, level, filename, lineno, message, (), None
            )
            record.extra_fields = kwargs
            self.logger.handle(record)
        else:
            # kwargs가 없는 경우에도 호출 스택 정보로 record 생성
            record = self.logger.makeRecord(
                self.name, level, filename, lineno, message, (), None
            )
            self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """DEBUG 레벨 로그"""
        self._log_with_extra(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """INFO 레벨 로그"""
        self._log_with_extra(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """WARNING 레벨 로그"""
        self._log_with_extra(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """ERROR 레벨 로그"""
        self._log_with_extra(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """CRITICAL 레벨 로그"""
        self._log_with_extra(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """예외 정보와 함께 로그 기록 (자동으로 traceback 포함)"""
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.ERROR, "", 0, message, (), None
            )
            record.extra_fields = kwargs
            self.logger.handle(record)
        else:
            self.logger.exception(message)
    
    def log_dict(self, level: int, data: Dict[str, Any], message: str = ""):
        """딕셔너리 데이터를 로그로 기록"""
        if message:
            self._log_with_extra(level, f"{message}: {json.dumps(data, ensure_ascii=False)}")
        else:
            self._log_with_extra(level, json.dumps(data, ensure_ascii=False))


def get_logger(
    name: str,
    level: Union[str, int] = None,
    **kwargs
) -> EstalanLogger:
    """
    EstalanLogger 인스턴스를 반환하는 팩토리 함수
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        level: 로그 레벨 (환경변수 ESTALAN_LOG_LEVEL에서 가져옴)
        **kwargs: EstalanLogger 생성자에 전달할 추가 인자들
    
    Returns:
        EstalanLogger 인스턴스
    """
    # 환경변수에서 로그 레벨 가져오기
    if level is None:
        level = os.getenv("ESTALAN_LOG_LEVEL", "INFO")
    
    # 기본 설정
    default_kwargs = {
        'use_colors': os.getenv("ESTALAN_USE_COLORS", "true").lower() == "true",
        'use_json': os.getenv("ESTALAN_USE_JSON", "false").lower() == "true",
        'enable_console': os.getenv("ESTALAN_ENABLE_CONSOLE", "true").lower() == "true",
        'enable_file': os.getenv("ESTALAN_ENABLE_FILE", "false").lower() == "true",
        'log_file': os.getenv("ESTALAN_LOG_FILE", f"logs/{name.replace('.', '/')}.log"),
    }
    
    # 사용자 설정으로 덮어쓰기
    default_kwargs.update(kwargs)
    
    return EstalanLogger(name, level, **default_kwargs)


# 편의를 위한 함수들
def debug(message: str, **kwargs):
    """기본 로거로 DEBUG 로그"""
    logger = get_logger("estalan")
    logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """기본 로거로 INFO 로그"""
    logger = get_logger("estalan")
    logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """기본 로거로 WARNING 로그"""
    logger = get_logger("estalan")
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """기본 로거로 ERROR 로그"""
    logger = get_logger("estalan")
    logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """기본 로거로 CRITICAL 로그"""
    logger = get_logger("estalan")
    logger.critical(message, **kwargs)


def exception(message: str, **kwargs):
    """기본 로거로 예외 로그"""
    logger = get_logger("estalan")
    logger.exception(message, **kwargs)
