"""日志：UTF-8 控制台 + 按大小轮转文件。Windows 控制台默认 GBK，中文会乱码，这里强制 UTF-8。"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("policymind")
    root.setLevel(settings.log_level.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    try:
        sh.stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sh.setFormatter(fmt)
    root.addHandler(sh)

    try:
        Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            Path(settings.log_dir) / "app.log",
            maxBytes=settings.log_rotation_mb * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"policymind.{name}")
