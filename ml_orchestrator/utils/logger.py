import logging
from pathlib import Path


def get_logger(log_path: Path) -> logging.Logger:
    """Return an append-only file logger for a project run."""
    logger = logging.getLogger("ml_orchestrator")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
