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

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric value.
        """
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        classes = logits.argmax(dim=-1)
        self.classification_metric.update(classes, labels)
        result = self.classification_metric.compute()
        return float(result.detach().cpu().item())

