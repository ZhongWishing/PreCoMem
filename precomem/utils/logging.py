"""Logging and reproducibility helpers."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_ROOT_NAME = "precomem"


def get_logger(name: str = _ROOT_NAME, level: Optional[int] = None) -> logging.Logger:
    """Return a logger under the ``precomem`` namespace.

    A single stream handler is attached to the root ``precomem`` logger, so
    sub-loggers (``precomem.agent``, ``precomem.retrieval``, ...) inherit it
    and application code can adjust verbosity in one place with
    ``logging.getLogger("precomem").setLevel(...)``.
    """
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        root.propagate = False
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
