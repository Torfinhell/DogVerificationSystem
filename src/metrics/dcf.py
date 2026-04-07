import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.epoch_metric import EpochMetric


class DCFMetric(EpochMetric):
    """
    Minimum detection cost (NIST SRE style) from genuine vs impostor softmax scores.
    """

    def __init__(
        self,
        device,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = self._resolve_device(device)
        self.p_target = p_target
        self.c_miss = c_miss
        self.c_fa = c_fa
        self._genuine: list[float] = []
        self._impostor: list[float] = []

    def __call__(self, logits: torch.Tensor, label: torch.Tensor, **kwargs):
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
        return None

    def finalize(self) -> dict[str, float]:
        g = np.asarray(self._genuine, dtype=np.float64)
        i = np.asarray(self._impostor, dtype=np.float64)
        if g.size == 0 or i.size == 0:
            return {"minDCF": float("nan")}
        thresholds = np.sort(np.unique(np.concatenate([g, i])))[::-1]
        best = float("inf")
        for t in thresholds:
            fa = float(np.mean(i >= t))
            miss = float(np.mean(g < t))
            dcf = self.c_miss * miss * self.p_target + self.c_fa * fa * (
                1.0 - self.p_target
            )
            if dcf < best:
                best = dcf
        return {"minDCF": float(best)}

    def reset(self) -> None:
        self._genuine.clear()
        self._impostor.clear()
