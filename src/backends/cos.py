import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.backends.base import BaseBackend

logger = logging.getLogger(__name__)

class CosineBackend(BaseBackend):
    NAME = "COS"

    def __init__(self, labels, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.register_buffer("label_reference", torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels)
        self.prototypes = None
        self._is_fitted = False

    def reset(self):
        self.prototypes = None
        self._is_fitted = False

    def fit(self, embeddings, labels):
        prototypes = []
        for target_label in self.label_reference:
            mask = (labels == target_label)
            if mask.any():
                prototype = embeddings[mask].mean(dim=0)
            else:
                prototype = torch.zeros(embeddings.shape[1], device=embeddings.device)
            
            if self.normalize:
                prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)
            
        self.prototypes = torch.stack(prototypes)
        self._is_fitted = True

    def predict(self, embeddings):
        if not self._is_fitted:
            raise ValueError("Backend not fitted")
            
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        scores = torch.mm(embeddings, self.prototypes.t())
        indices = scores.argmax(dim=-1)
        return self.label_reference[indices]
