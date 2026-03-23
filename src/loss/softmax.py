import torch
from torch import nn


class SoftmaxLoss(nn.Module):
    """
    Implementation softmax loss. Commonly used for classification tasks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): predictions (Batch_speakers, NUM_SPEAKERS)
            labels (Tensor): ground-truth labels.(Batch_speakers, 1)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, labels)}
