"""Backend implementations for speaker verification."""

from src.backends.base import BaseBackend
from src.backends.cos import CosineBackend
from src.backends.plda import PLDABackend

__all__ = [
    "BaseBackend",
    "CosineBackend",
    "PLDABackend",
]
