"""
Structured Logging for UIDAI Governance Intelligence Framework.

Implements JSON-formatted logging for production environments
with context-aware logging and performance tracking.
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class PerformanceLogger:
    """Track and log performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers: Dict[str, float] = {}
    
    @contextmanager
    def timer(self, operation: str, **context):
        """Context manager for timing operations."""
        start_time = time.time()
        self.logger.info(
            f"Starting {operation}",
            extra={"operation": operation, "event": "start", **context}
        )
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed {operation}",
                extra={
                    "operation": operation,
                    "event": "error",
                    "duration_seconds": duration,
                    "error": str(e),
                    **context
                },
                exc_info=True
            )
            raise
        else:
            duration = time.time() - start_time
            self.logger.info(
                f"Completed {operation}",
                extra={
                    "operation": operation,
                    "event": "complete",
                    "duration_seconds": duration,
                    **context
                }
            )


def setup_logger(
    name: str,
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Path] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup a structured logger with JSON formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional file path for logging
        context: Optional context dictionary to add to all logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={
                'asctime': 'timestamp',
                'levelname': 'level',
                'name': 'logger'
            }
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add context filter
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get or create a logger with optional configuration.
    
    If logger already exists, returns it. Otherwise creates a new one.
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        # Try to import config, fall back to defaults
        try:
            from config_manager import config
            level = level or config.logging.level
            log_format = log_format or config.logging.format
            log_file = log_file or config.logging.log_file
        except ImportError:
            level = level or "INFO"
            log_format = log_format or "json"
        
        logger = setup_logger(name, level, log_format, log_file, context)
    
    return logger


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """Log information about a pandas DataFrame."""
    logger.info(
        f"{name} info",
        extra={
            "dataframe": name,
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict()
        }
    )


# Example usage and convenience functions
def log_pipeline_start(logger: logging.Logger, pipeline_name: str, **kwargs):
    """Log the start of a data pipeline."""
    logger.info(
        f"Pipeline started: {pipeline_name}",
        extra={
            "pipeline": pipeline_name,
            "event": "pipeline_start",
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
    )


def log_pipeline_end(logger: logging.Logger, pipeline_name: str, success: bool = True, **kwargs):
    """Log the end of a data pipeline."""
    logger.info(
        f"Pipeline {'completed' if success else 'failed'}: {pipeline_name}",
        extra={
            "pipeline": pipeline_name,
            "event": "pipeline_end",
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
    )


def log_model_metrics(logger: logging.Logger, model_name: str, metrics: Dict[str, float]):
    """Log model performance metrics."""
    logger.info(
        f"Model metrics: {model_name}",
        extra={
            "model": model_name,
            "event": "model_metrics",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Made with Bob
