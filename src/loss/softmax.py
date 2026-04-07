import torch
from torch import nn


class SoftmaxLoss(nn.Module):
    """
    Implementation softmax loss. Commonly used for classification tasks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, label: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): predictions (Batch_speakers, NUM_SPEAKERS)
            label (Tensor): ground-truth label.(Batch_speakers,)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, label)}
