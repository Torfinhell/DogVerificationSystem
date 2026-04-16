import numpy as np
import torch
import torch.nn.functional as F
from .base_metric import BaseMetric
from tqdm.auto import tqdm

class EERMetric(BaseMetric):
    """
    Equal error rate from genuine vs impostor softmax scores (multiclass verification view).
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
        self._genuine: list[float] = []
        self._impostor: list[float] = []

    def update(self, **batch):
        """
        Update internal genuine/impostor score lists.

        Accepts either:
            - backend_scores or cos_scores + label (pairwise similarity matrix)
            - logits + label (classification logits)
        """
        logits = batch.get("logits")
        label = batch.get("label")
        logits = logits.to(self.device)
        label = label.to(self.device)
        probs = F.softmax(logits, dim=-1)
        label = label.long()
        b = probs.shape[0]
        idx = torch.arange(b, device=probs.device)
        self._genuine.extend(probs[idx, label].detach().cpu().numpy().tolist())
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[idx, label] = False
        self._impostor.extend(probs[mask].detach().cpu().numpy().tolist())
    def compute(self, **batch):
        """
        Compute the Equal Error Rate from the collected genuine and impostor scores.
        Returns a dict with key "EER".
        """
        g = np.asarray(self._genuine, dtype=np.float64)
        i = np.asarray(self._impostor, dtype=np.float64)
        if g.size == 0 or i.size == 0:
            return float("nan")
        thresholds = np.sort(np.unique(np.concatenate([g, i])))[::-1]
        best_diff = float("inf")
        eer = 0.0
        for t in tqdm(thresholds, desc="Running thresholds for EER"):
            fa = float(np.mean(i >= t))
            miss = float(np.mean(g < t))
            diff = abs(fa - miss)
            if diff < best_diff:
                best_diff = diff
                eer = (fa + miss) / 2.0
        return float(eer)

    def reset(self, **batch):
        """Clear the stored genuine and impostor score lists."""
        self._genuine.clear()
        self._impostor.clear()

