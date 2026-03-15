import logging
from os import error
import sys
from pathlib import Path
from src.config import settings

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("diabetes_prediction")

# Avoid adding duplicate handlers if called multiple times
if not logger.handlers:
    logger.setLevel(logging.DEBUG if settings.app_env == "development" else logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_formatter = logging.Formatter(
        fmt="%(asctime)s | %(filename)s:%(lineno)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1 — print to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Handler 2 — write everything to app.log
    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Handler 3 — write only errors to errors.log
    error_handler = logging.FileHandler(LOG_DIR / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(error_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

# Now you can import `logger` from this module and use it directly
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.error("This is an error message")
