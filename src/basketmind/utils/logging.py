"""Logging setup shared by every pipeline stage."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the 'basketmind' namespace with one console handler."""
    logger = logging.getLogger(f"basketmind.{name}")
    root = logging.getLogger("basketmind")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(level)
    return logger
