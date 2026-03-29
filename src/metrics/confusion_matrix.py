import torch
from torchmetrics.classification import ConfusionMatrix
from src.metrics.base_metric import BaseMetric

class ConfusionMatrixMetric(BaseMetric):
    def __init__(self, num_classes, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ConfusionMatrix(
            num_classes=num_classes,
            task="multiclass"
        ).to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        classes = logits.argmax(dim=-1)
        result = self.metric(classes, labels)
        return result.detach().cpu()