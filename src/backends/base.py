"""Base backend class for speaker verification."""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseBackend(ABC, nn.Module):
    """
    Abstract base class for all backends used in evaluation/inference.
    
    Backends implement different similarity computation methods
    for speaker/dog verification tasks.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, embeddings1, embeddings2):
        """Forward pass for the backend."""
        pass
