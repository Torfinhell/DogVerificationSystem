from abc import ABC, abstractmethod

import torch

from src.metrics.base_metric import BaseMetric


class EpochMetric(BaseMetric, ABC):
    """
    Metrics that aggregate over an entire eval pass and expose a scalar dict
    via ``finalize()`` once per epoch (not averaged per batch in MetricTracker).
    Trainers detect these via ``isinstance(..., EpochMetric)``.
    """

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @abstractmethod
    def finalize(self) -> dict[str, float]:
        """Return metric name(s) and value(s) after all batches have been seen."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before the next epoch or partition."""
