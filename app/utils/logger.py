"""
Centralized logging configuration for YouTube RAG Assistant.
All modules import get_logger() from here instead of using print().
"""

import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with consistent formatting.

    Usage in any module:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.warning("Metadata fetch failed, using fallback")
        logger.error("Groq API call failed: %s", str(e))
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/rag_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
