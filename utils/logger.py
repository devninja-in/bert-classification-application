import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(name, log_file=LOG_FILE, level=LOG_LEVEL):
    """
    Configure logger with both console and file handlers

    Args:
        name (str): Logger name
        log_file (Path): Path to log file
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create main application logger
logger = setup_logger("bert_classifier")