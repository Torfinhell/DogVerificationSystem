import torch

from src.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    def __init__(self, classification_metric, device, *args, **kwargs):
        """
        Classification metric wrapper for TorchMetrics.
        Ensures metric is on correct device and handles state properly.

        Args:
            classification_metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.classification_metric = classification_metric.to(device)
        self._num_classes = getattr(classification_metric, 'num_classes', None)

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        """Update num_classes for the underlying torchmetrics object if possible."""
        self._num_classes = value
        if hasattr(self.classification_metric, 'num_classes'):
            # Try to update num_classes, but some torchmetrics don't support this
            try:
                self.classification_metric.num_classes = value
            except (AttributeError, ValueError):
                # If we can't update, we'll need to recreate the metric
                pass

    def __call__(self, logits: torch.Tensor, label: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            label (Tensor): ground-truth label.
        Returns:
            metric (float): calculated metric value.
        """
        logits = logits.to(self.device)
        label = label.to(self.device)
        classes = logits.argmax(dim=-1)
        self.classification_metric.update(classes, label)
        result = self.classification_metric.compute()
        return float(result.detach().cpu().item())

