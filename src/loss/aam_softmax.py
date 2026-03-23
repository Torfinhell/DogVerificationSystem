import torch
from torch import nn
import torch.nn.functional as F

class AAMSoftmaxLoss(nn.Module):
    """
    Implementation AAM-softmax loss. Commonly used for speaker verification tasks
    """

    def __init__(self, embedding_dim, num_speakers, scale, margin):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.margin=margin
        self.scale=scale
        self.embedding=nn.Embedding(num_speakers, embedding_dim, max_norm=1)
        self.num_speakers=num_speakers

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): predictions (Batch_speakers, NUM_SPEAKERS)
            labels (Tensor): ground-truth labels.(Batch_speakers,)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        b, ns=logits.shape
        assert b==len(labels) and ns==self.num_speakers
        logits = F.normalize(logits, dim=1)
        cos_theta = torch.matmul(self.embedding.weight, logits.T).T
        psi = cos_theta - self.margin
        onehot = F.one_hot(labels, self.num_speakers)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        return {"loss": self.loss(logits, labels)}

