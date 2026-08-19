"""Consistent human-readable and file-based operational logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(log_dir: Path | None = None, verbosity: int = 0) -> logging.Logger:
    """Configure the platform logger once and return its named instance."""
    level = logging.DEBUG if verbosity else logging.INFO
    logger = logging.getLogger("anomdet")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = RichHandler(rich_tracebacks=True, show_path=False, console=None)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "platform.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger("py.warnings").addHandler(logging.NullHandler())
    return logger


def log_exception(logger: logging.Logger, context: str) -> None:
    """Record a command failure with a concise, searchable context message."""
    logger.exception("Command failed while %s", context)
