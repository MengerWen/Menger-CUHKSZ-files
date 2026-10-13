"""Shared logging configuration for Python CLI commands."""

from __future__ import annotations

import logging



def configure_logging(level: str = "INFO") -> None:
    """Configure a simple, readable root logger for command execution."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )