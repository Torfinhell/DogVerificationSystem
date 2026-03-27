import torch

from src.metrics.base_metric import BaseMetric


class ConfusionMatrixMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        result = self.metric(classes, labels)

        if result is None and hasattr(self.metric, "compute"):
            result = self.metric.compute()
            self.metric.reset()

        if isinstance(result, torch.Tensor):
            if result.ndim == 2:
                return result.detach().cpu()
            if result.numel() == 1:
                return float(result.detach().cpu().item())

        return result
