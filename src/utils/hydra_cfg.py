"""
Helpers for reading Hydra ``DictConfig`` / nested dicts without ``OmegaConf`` in app code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from omegaconf import DictConfig

def cfg_to_container(cfg: Any) -> Any:
    """Recursively convert a Hydra config node to plain dict/list/scalars (e.g. for W&B).
    
    Skips unresolvable interpolations to avoid ConfigErrors during conversion.
    """
    if cfg is None or isinstance(cfg, (bool, int, float, str)):
        return cfg
    if isinstance(cfg, DictConfig):
        result = {}
        for k, v in cfg.items_ex(resolve=False):
            try:
                # Try to resolve with interpolation
                resolved = cfg[k]
                result[str(k)] = cfg_to_container(resolved)
            except Exception:
                # If interpolation fails, just use the raw value
                result[str(k)] = cfg_to_container(v)
        return result
    if isinstance(cfg, Mapping):
        return {str(k): cfg_to_container(v) for k, v in cfg.items()}
    if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        return [cfg_to_container(x) for x in cfg]
    return cfg
