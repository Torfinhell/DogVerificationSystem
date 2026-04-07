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
        self.prototypes = None
        self._is_fitted = False

    def fit(self, embeddings, labels):
        """Fit cosine backend by computing class prototypes."""
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            mask = labels == label
            prototype = embeddings[mask].mean(dim=0)
            if self.normalize:
                prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)
        self.prototypes = torch.stack(prototypes)
        self._is_fitted = True

    def predict(self, embeddings):
        """Predict class labels using cosine similarity to prototypes."""
        if not self._is_fitted:
            raise ValueError("Backend not fitted")
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = torch.mm(embeddings, self.prototypes.t())
        return scores.argmax(dim=-1)

    def forward(self, embeddings1, embeddings2):
        if self.normalize:
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        scores = torch.mm(embeddings1, embeddings2.t())
        return scores
