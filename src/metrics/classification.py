import torch

from src.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Classification metric wrapper for TorchMetrics.
        Ensures metric is on correct device and handles state properly.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric value.
        """
        # Ensure tensors are on the correct device
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        classes = logits.argmax(dim=-1)
        
        # Update metric with batch data (accumulates internally)
        self.metric.update(classes, labels)
        
        # Compute result (torchmetrics computes on accumulated state)
        result = self.metric.compute()
        
        # Don't reset - let metric accumulate across batches in epoch
        # Reset happens in MetricTracker when epoch ends
        
        if isinstance(result, torch.Tensor):
            if result.numel() == 1:
                return float(result.detach().cpu().item())
            else:
                return result.detach().cpu()

        return float(result) if result is not None else 0.0

