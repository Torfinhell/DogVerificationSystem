import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.epoch_metric import EpochMetric


class EERMetric(EpochMetric):
    """
    Equal error rate from genuine vs impostor softmax scores (multiclass verification view).
    """

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self._resolve_device(device)
        self._genuine: list[float] = []
        self._impostor: list[float] = []

    def __call__(self, logits: torch.Tensor = None, label: torch.Tensor = None, **kwargs):
        scores_key = None
        if "backend_scores" in kwargs:
            scores_key = "backend_scores"
        elif "cos_scores" in kwargs:
            scores_key = "cos_scores"
        
        if scores_key is not None:
            scores = kwargs[scores_key].to(self.device)
            label = label.to(self.device)
            b = label.shape[0]
            for i in range(b):
                for j in range(b):
                    if i == j:
                        continue
                    score = scores[i, j].item()
                    if label[i] == label[j]:
                        self._genuine.append(score)  
                    else:
                        self._impostor.append(score) 
        elif logits is not None and label is not None:
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
            return {"EER": float("nan")}
        thresholds = np.sort(np.unique(np.concatenate([g, i])))[::-1]
        best_diff = float("inf")
        eer = 0.0
        for t in thresholds:
            fa = float(np.mean(i >= t))
            miss = float(np.mean(g < t))
            diff = abs(fa - miss)
            if diff < best_diff:
                best_diff = diff
                eer = (fa + miss) / 2.0
        return {"EER": float(eer)}

    def reset(self) -> None:
        self._genuine.clear()
        self._impostor.clear()
