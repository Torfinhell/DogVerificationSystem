import torch
from torch import nn
import torch.nn.functional as F

class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, scale, margin, label_smoothing, labels):
        super().__init__()
        self.labels = list(labels)
        self.num_speakers = len(self.labels)
        self.embedding_dim = embedding_dim
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(self.num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.label_to_ind = {label: i for i, label in enumerate(self.labels)}
        self.ind_to_label = {i: label for i, label in enumerate(self.labels)}

    def forward(self, logits: torch.Tensor, label, **batch):
        b, ns = logits.shape
        
        if torch.is_tensor(label):
            label_list = label.tolist()
        else:
            label_list = label
            
        mapped_label = torch.tensor(
            [self.label_to_ind[l] for l in label_list], 
            device=logits.device, 
            dtype=torch.long
        )

        assert b == len(mapped_label) and ns == self.embedding_dim
        
        normalized_logits = F.normalize(logits, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)
        cos_theta = torch.matmul(normalized_logits, normalized_weight.T)
        
        psi = cos_theta - self.margin
        onehot = F.one_hot(mapped_label, self.num_speakers)
        output_logits = self.scale * torch.where(onehot == 1, psi, cos_theta)  
        
        pred_indices = output_logits.argmax(dim=-1)
        preds = [self.ind_to_label[idx.item()] for idx in pred_indices]
        
        if len(preds) > 0 and isinstance(preds[0], (int, float)):
            preds = torch.tensor(preds, device=logits.device)

        return {
            "loss": self.loss(output_logits, mapped_label), 
            "logits": output_logits, 
            "pred": preds
        }
