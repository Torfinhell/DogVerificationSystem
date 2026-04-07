import torch
from torch import nn
import torch.nn.functional as F

class AAMSoftmaxLoss(nn.Module):
    """
    Implementation AAM-softmax loss. Commonly used for speaker verification tasks
    """

    def __init__(self, embedding_dim, num_speakers, scale, margin, label_smoothing):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.margin=margin
        self.scale=scale
        self.embedding=nn.Embedding(num_speakers, embedding_dim, max_norm=1)
        self.num_speakers=num_speakers
        self.embedding_dim=embedding_dim

    def forward(self, logits: torch.Tensor, label: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): predictions (Batch_speakers, NUM_SPEAKERS)
            label (Tensor): ground-truth label.(Batch_speakers,)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        b, ns=logits.shape
        assert b==len(label) and ns==self.embedding_dim
        logits = F.normalize(logits, dim=1)
        cos_theta = torch.matmul(self.embedding.weight, logits.T).T
        psi = cos_theta - self.margin
        onehot = F.one_hot(label, self.num_speakers)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        return {"loss": self.loss(logits, label), "logits":logits}

