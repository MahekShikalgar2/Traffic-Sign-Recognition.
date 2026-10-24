"""
Logging configuration for the Traffic Sign Recognition application.
Provides a rotating file handler and a stream handler.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> None:
    """
    Initialise the root logger with:
      - RotatingFileHandler  → logs/app.log  (2 MB, 3 backups)
      - StreamHandler        → stderr
    Call once at application startup.
    """
    if log_dir is None:
        # Default: logs/ next to the workspace root
        log_dir = Path(__file__).parent.parent.parent / 'logs'

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'app.log'

    root_logger = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,  # 2 MB
        backupCount=3,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Route Python warnings through the logging system
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call setup_logging() first."""
    return logging.getLogger(name)
