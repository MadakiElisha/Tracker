#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Structured Logger

 Single module that all other files import for logging.
 Replaces scattered print() calls with structured, levelled,
 timestamped log output that goes to both console and file.

 Usage in any module:
   from tracker_logger import get_logger
   log = get_logger(__name__)
   log.info("Connected to FC")
   log.warning("Target lost for 30 frames")
   log.error("Telemetry thread exception: %s", e)
=============================================================
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module name.
    All loggers share the same root handler configuration.
    Call setup_logging() once at startup before using any logger.
    """
    return logging.getLogger(name)


def setup_logging(cfg: dict) -> logging.Logger:
    """
    Configure the root logger from config dict.
    Call this exactly once at the top of tracker.py main().

    cfg: the logging section from config.yaml
    """
    level_str = cfg.get('level', 'INFO').upper()
    level     = getattr(logging, level_str, logging.INFO)

    # Custom formatter with millisecond timestamp
    fmt = logging.Formatter(
        fmt     = '%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)-20s %(message)s',
        datefmt = '%H:%M:%S'
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers (prevents duplicate output on reload)
    root.handlers.clear()

    # Console handler
    if cfg.get('console', True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(level)
        root.addHandler(ch)

    # File handler (rotating)
    log_file = cfg.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            filename    = str(log_path),
            maxBytes    = cfg.get('max_bytes', 5_242_880),
            backupCount = cfg.get('backup_count', 3),
            encoding    = 'utf-8',
        )
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)

    root.info("Logging initialised — level=%s file=%s", level_str, log_file or "none")
    return root


class ExceptionCounter:
    """
    Counts exceptions per named category.
    Used to surface error rates in the HUD without crashing.

    Usage:
        ec = ExceptionCounter()
        try:
            ...
        except Exception as e:
            ec.record('telemetry', e)

    Retrieve for HUD:
        ec.summary()   → {'telemetry': 3, 'velocity': 0}
    """

    def __init__(self):
        self._counts: dict = {}
        self._last:   dict = {}
        self._log = get_logger(self.__class__.__name__)

    def record(self, category: str, exc: Exception,
               log_level: int = logging.WARNING):
        self._counts[category] = self._counts.get(category, 0) + 1
        msg = f"[{category}] #{self._counts[category]}: {type(exc).__name__}: {exc}"
        self._log.log(log_level, msg)
        self._last[category] = str(exc)

    def count(self, category: str) -> int:
        return self._counts.get(category, 0)

    def last(self, category: str) -> str:
        return self._last.get(category, '')

    def summary(self) -> dict:
        return dict(self._counts)

    def total(self) -> int:
        return sum(self._counts.values())
