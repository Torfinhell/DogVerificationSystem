import numpy as np
import torch
import torch.nn.functional as F
import inspect

from src.metrics.base_metric import BaseMetric
from tqdm.auto import tqdm

class DCFMetric(BaseMetric):
    """
    Minimum detection cost (NIST SRE style) from genuine vs impostor softmax scores.
    """

    def __init__(
        self,
        device,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
        name=None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.p_target = p_target
        self.c_miss = c_miss
        self.c_fa = c_fa
        self._genuine: list[float] = []
        self._impostor: list[float] = []

    def update(self, **batch):
        """
        Update internal genuine/impostor score lists from logits and labels.
        """
        logits = batch.get("logits")
        label = batch.get("label")
        if logits is None or label is None:
            raise ValueError("DCFMetric.update() requires 'logits' and 'label' in batch")

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
        Compute the minimum Detection Cost Function (minDCF).
        Returns a dict with key "minDCF".
        """
        g = np.asarray(self._genuine, dtype=np.float64)
        i = np.asarray(self._impostor, dtype=np.float64)
        if g.size == 0 or i.size == 0:
            return float("nan")

        thresholds = np.sort(np.unique(np.concatenate([g, i])))[::-1]
        best = float("inf")
        for t in tqdm(thresholds, desc="Running thresholds for EER"):
            fa = float(np.mean(i >= t))
            miss = float(np.mean(g < t))
            dcf = self.c_miss * miss * self.p_target + self.c_fa * fa * (1.0 - self.p_target)
            if dcf < best:
                best = dcf
        return  float(best)

    def reset(self, **batch):
        """Clear the stored genuine and impostor score lists."""
        self._genuine.clear()
        self._impostor.clear()


