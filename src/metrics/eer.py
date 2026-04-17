import torch
import torch.nn.functional as F
from .base_metric import BaseMetric
from torchmetrics.classification import BinaryEER


class EERMetric(BaseMetric):
    """
    Equal error rate from genuine vs impostor softmax scores (verification task).
    Optimized using torchmetrics.BinaryEER for efficient threshold evaluation.
    """

    def __init__(self, device, name=None, *args, **kwargs):
        """
        Args:
            device (torch.device or str): device to use for tensor computations.
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(name=name, *args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        
        self.eer_metric = BinaryEER(thresholds=None).to(self.device)

    def update(self, **batch):
        """
        Update internal genuine/impostor score lists.

        Args:
            batch (dict): Contains 'logits' and 'label' tensors
        """
        logits = batch.get("logits")
        label = batch.get("label")
        logits = logits.to(self.device)
        label = label.to(self.device)
        probs = F.softmax(logits, dim=-1)
        label = label.long()
        b = probs.shape[0]
        idx = torch.arange(b, device=probs.device)
        
        # Extract genuine scores (diagonal)
        genuine = probs[idx, label]
        
        # Extract impostor scores (off-diagonal)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[idx, label] = False
        impostor = probs[mask]
        
        # Use torchmetrics: genuine = 1, impostor = 0
        preds = torch.cat([genuine, impostor])
        targets = torch.cat([
            torch.ones(genuine.shape[0], device=self.device, dtype=torch.long),
            torch.zeros(impostor.shape[0], device=self.device, dtype=torch.long)
        ])
        self.eer_metric.update(preds, targets)

    def compute(self, **batch):
        """
        Compute the Equal Error Rate from collected scores.
        
        Returns:
            float: EER value between 0 and 1
        """
        eer = self.eer_metric.compute()
        return float(eer)

    def reset(self, **batch):
        """Clear stored genuine and impostor scores."""
        self.eer_metric.reset()



