import numpy as np
import torch
import torch.nn.functional as F
from src.metrics.base_metric import BaseMetric

class ClassificationMetric(BaseMetric):
    """
    Wrapper for torchmetrics classification metrics.
    Handles mapping of arbitrary labels to 0..N-1 indices.
    """

    def __init__(self, classification_metric, labels, device, name=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.classification_metric = classification_metric.to(self.device)
        self.label_to_ind = {label: i for i, label in enumerate(labels)}
    def update(self, **batch):
        """
        Update the wrapped torchmetrics metric.
        Maps 'pred' and 'label' to indices before updating.
        """
        assert "pred" in batch and "label" in batch
        preds = torch.tensor([self.label_to_ind[int(val)] for val in batch["pred"]], device=self.device)
        labels = torch.tensor([self.label_to_ind[int(val)] for val in batch["label"]], device=self.device)
        self.classification_metric.update(preds, labels)

    def compute(self, **batch):
        """Returns the computed metric as a float."""
        result = self.classification_metric.compute()
        return result.detach().cpu().item()

    def reset(self, **batch):
        """Reset the internal torchmetrics state."""
        self.classification_metric.reset()
