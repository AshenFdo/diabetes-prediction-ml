import logging
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Log file — a new file is created on every run (timestamped name)
# ---------------------------------------------------------------------------
_LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
_LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(_LOGS_DIR, _LOG_FILE)


# ---------------------------------------------------------------------------
# Custom formatter — two formats depending on log level
#   Normal  (DEBUG / INFO / WARNING) : timestamp | level | logger | message
#   Error   (ERROR / CRITICAL)       : timestamp | file:line | func | level | message
# ---------------------------------------------------------------------------
class LevelBasedFormatter(logging.Formatter):
    _NORMAL_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _ERROR_FMT  = (
        "%(asctime)s | %(filename)s:%(lineno)d | %(funcName)s | "
        "%(levelname)s | %(message)s"
    )
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            self._style._fmt = self._ERROR_FMT
        else:
            self._style._fmt = self._NORMAL_FMT
        self.datefmt = self._DATEFMT
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger factory — called once; subsequent imports reuse the same logger
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("DiabetesPrediction")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # prevent duplicate handlers on re-import
        return logger

    formatter = LevelBasedFormatter()

    # File handler — captures everything (DEBUG and above)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler — shows INFO and above in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = _build_logger()


if __name__ == "__main__":
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
