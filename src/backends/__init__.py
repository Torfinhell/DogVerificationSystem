"""Backend implementations for speaker verification."""

from src.backends.base import BaseBackend
from src.backends.cos import CosineBackend
from src.backends.plda import PLDABackend
from src.backends.mlp import MLPBackend

__all__ = [
    "BaseBackend",
    "CosineBackend",
    "PLDABackend",
    "MLPBackend",
]
