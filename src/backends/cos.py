"""Cosine similarity backend for speaker/dog verification."""

import torch
import torch.nn.functional as F

from src.backends.base import BaseBackend


class CosineBackend(BaseBackend):
    """
    Cosine similarity backend.
    
    Computes cosine similarity between embeddings for verification tasks.
    This is a simple baseline backend that normalizes embeddings and
    computes their dot product similarity.
    """

    def __init__(self, normalize=True):
        """
        Initialize the Cosine backend.
        
        Args:
            normalize (bool): Whether to L2-normalize embeddings before
                computing similarity. Default: True
        """
        super().__init__()
        self.normalize = normalize
    def forward(self, embeddings1, embeddings2):
        if self.normalize:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        scores = torch.mm(embeddings1, embeddings2.t())
        return scores
