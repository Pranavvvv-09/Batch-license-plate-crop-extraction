"""Rotating file + console logging, configured once at process start."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import load

_configured = False


def setup() -> logging.Logger:
    global _configured
    logger = logging.getLogger("sang_det")
    if _configured:
        return logger

    cfg = load()
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_path = cfg.path("logging.file")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=int(cfg.get("logging.max_bytes", 10 * 1024 * 1024)),
            backupCount=int(cfg.get("logging.backup_count", 5)),
            encoding="utf-8",
        )
        rotating.setFormatter(fmt)
        logger.addHandler(rotating)
    except OSError as exc:  # console-only is still a usable run
        logger.warning("File logging disabled (%s)", exc)

    # Ultralytics/yt-dlp are chatty at INFO; keep our log readable.
    for noisy in ("ultralytics", "yt_dlp", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return logger


def get(name: str = "") -> logging.Logger:
    base = setup()
    return base.getChild(name) if name else base
