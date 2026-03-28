"""
Helpers for reading Hydra ``DictConfig`` / nested dicts without ``OmegaConf`` in app code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Dotted lookup, e.g. ``writer.log_confusion_matrix_image``."""
    node = cfg
    for part in key.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(part)
        elif hasattr(node, "get"):
            node = node.get(part)
        else:
            return default
    return default if node is None else node


def cfg_to_container(cfg: Any) -> Any:
    """Recursively convert a Hydra config node to plain dict/list/scalars (e.g. for W&B)."""
    if cfg is None or isinstance(cfg, (bool, int, float, str)):
        return cfg
    if isinstance(cfg, Mapping):
        return {str(k): cfg_to_container(v) for k, v in cfg.items()}
    if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        return [cfg_to_container(x) for x in cfg]
    return cfg
