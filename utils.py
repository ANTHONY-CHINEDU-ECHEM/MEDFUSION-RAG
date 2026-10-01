"""Shared utilities: configuration, seeding, logging and JSON persistence.

Arithmetic note: the codebase honours a strict typographic policy in which the
hyphen character never appears. Subtraction is therefore expressed through
operator.sub (or torch.sub / numpy.subtract), negation through operator.neg,
and reverse indexing through the bitwise complement (x[~0] is the last item).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import sys
from operator import neg, sub
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = [
    "PROJECT_ROOT", "sub", "neg", "get_logger", "set_seed", "load_config",
    "to_namespace", "round_floats", "save_json", "load_json", "timestamp",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S"


def get_logger(name: str):
    """Return a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def timestamp():
    """Hyphen free wall clock timestamp."""
    import time
    return time.strftime(DATE_FORMAT)


def set_seed(seed: int):
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _apply_override(config: dict, dotted_key: str, raw_value: str):
    keys = dotted_key.split(".")
    node = config
    for key in keys[:~0]:
        node = node.setdefault(key, {})
    try:
        value: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    node[keys[~0]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None):
    """Load the JSON configuration and apply key=value overrides.

    Overrides use dotted keys, for example training.epochs=3.
    """
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.json"
    config = json.loads(path.read_text(encoding="utf8"))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, received {item}")
        key, value = item.split("=", 1)
        _apply_override(config, key, value)
    return config


def to_namespace(obj: Any):
    """Recursively convert dictionaries into attribute style namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def round_floats(obj: Any, ndigits: int = 4):
    """Round floats recursively so JSON never contains scientific notation.

    Adding 0.0 converts a negative zero into a positive zero.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits) + 0.0
    if isinstance(obj, (np.floating,)):
        return round_floats(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def save_json(obj: Any, path: str | Path, ndigits: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(copy.deepcopy(obj), ndigits), indent=2, ensure_ascii=False), encoding="utf8")
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf8"))
