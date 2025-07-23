"""
Logging configuration for the Alan system.
"""

import logging
import logging.config
import os
import sys
from typing import Any


def get_logging_config(
    log_level: str = "INFO",
    log_format: str = None,
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = False,
) -> dict[str, Any]:
    """
    Generate logging configuration dictionary.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Path to log file (if file logging is enabled)
        enable_console: Enable console logging
        enable_file: Enable file logging

    Returns:
        Logging configuration dictionary
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "[%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {},
        "loggers": {
            "alan": {
                "level": log_level,
                "handlers": [],
                "propagate": False,
            },
            "alan.core": {
                "level": log_level,
                "handlers": [],
                "propagate": True,
            },
            "alan.tools": {
                "level": log_level,
                "handlers": [],
                "propagate": True,
            },
            "httpx": {
                "level": "WARNING",  # 외부 library는 Warning으로 고정.
                "handlers": [],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": [],
        },
    }

    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "stream": sys.stdout,
        }
        config["loggers"]["alan"]["handlers"].append("console")
        config["root"]["handlers"].append("console")

    # File handler
    if enable_file and log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        config["loggers"]["alan"]["handlers"].append("file")
        config["root"]["handlers"].append("file")

    return config


def setup_logging(
    log_level: str = None,
    log_format: str = None,
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = False,
) -> None:
    """
    Setup logging configuration for the Alan system.

    Args:
        log_level: Logging level from environment or default to INFO
        log_format: Custom log format string
        log_file: Path to log file (defaults to logs/alan.log if file logging enabled)
        enable_console: Enable console logging
        enable_file: Enable file logging from environment or default
    """
    # Get configuration from environment variables
    log_level = log_level or os.getenv("ALAN_LOG_LEVEL", "ERROR").upper()
    enable_file = (
        enable_file or os.getenv("ALAN_ENABLE_FILE_LOGGING", "false").lower() == "true"
    )
    log_file = log_file or os.getenv("ALAN_LOG_FILE", "logs/alan.log")

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"

    # Generate and apply configuration
    config = get_logging_config(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_console=enable_console,
        enable_file=enable_file,
    )

    logging.config.dictConfig(config)

    # Log setup completion
    logger = logging.getLogger("alan.setup")
    logger.info(
        f"Logging configured - Level: {log_level}, Console: {enable_console}, File: {enable_file}"
    )
    if enable_file:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Module-level function to ensure logging is set up
def ensure_logging_setup():
    """Ensure logging is set up if not already configured."""
    if not logging.getLogger("alan").handlers:
        setup_logging()


# Auto-setup logging when module is imported
ensure_logging_setup()
