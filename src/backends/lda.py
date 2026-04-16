import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.backends.base import BaseBackend

logger = logging.getLogger(__name__)

class PLDABackend(BaseBackend):
    NAME = "PLDA"

    def __init__(self, labels, n_components=256, n_iter=10, reg=1e-6, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.reg = reg
        self.config = kwargs
        
        self.register_buffer("label_reference", torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels)
        self.register_buffer("mean", None)  
        self.register_buffer("F", None)     
        self.register_buffer("G", None)     
        self.register_buffer("Sigma", None) 
        self._is_fitted = False        
        self.train_embeddings = None
        self.train_labels = None        

    def reset(self):
        self.mean = None
        self.F = None
        self.G = None
        self.Sigma = None
        self.train_embeddings = None
        self.train_labels = None
        self._is_fitted = False

    def predict(self, embeddings):
        if not self._is_fitted:
            raise ValueError("Backend not fitted")
        
        normalized_embeddings = embeddings - self.mean
        normalized_train = self.train_embeddings - self.mean
        
        scores = torch.mm(normalized_embeddings, normalized_train.t())
        _, indices = scores.max(dim=-1)
        
        return self.train_labels[indices]

    def fit(self, embeddings, labels=None):
        self.reset()
        
        if labels is not None:
            fit_labels = []
            fit_embeddings = []
            for i, target_label in enumerate(self.label_reference):
                mask = (labels == target_label)
                if mask.any():
                    fit_embeddings.append(embeddings[mask])
                    fit_labels.append(torch.full((mask.sum(),), target_label, device=embeddings.device))
            
            self.train_embeddings = torch.cat(fit_embeddings).detach().clone()
            self.train_labels = torch.cat(fit_labels).detach().clone()
        else:
            self.train_embeddings = embeddings.detach().clone()
            self.train_labels = None

        N, D = self.train_embeddings.shape
        device = self.train_embeddings.device
        mean = self.train_embeddings.mean(dim=0, keepdim=True)
        x = self.train_embeddings - mean
        
        if self.train_labels is None:
            cov = (x.t() @ x) / (N - 1)
            U, S, _ = torch.linalg.svd(cov)
            self.F = nn.Parameter(U[:, :self.n_components] @ torch.diag(torch.sqrt(S[:self.n_components])))
            self.G = nn.Parameter(self.F.clone() * 0.1)
            self.Sigma = nn.Parameter(torch.ones(D, device=device))
        else:
            unique_labels = torch.unique(self.train_labels)
            n_speakers = len(unique_labels)
            speaker_means = torch.stack([x[self.train_labels == l].mean(dim=0) for l in unique_labels])
            
            cov_between = (speaker_means.t() @ speaker_means) / n_speakers
            cov_within = torch.zeros(D, D, device=device)
            for l in unique_labels:
                x_s = x[self.train_labels == l]
                if len(x_s) > 1:
                    cov_within += (x_s.t() @ x_s) / (len(x_s) - 1)
            cov_within /= n_speakers
            
            U_b, S_b, _ = torch.linalg.svd(cov_between)
            U_w, S_w, _ = torch.linalg.svd(cov_within)
            
            n_comp = min(self.n_components, D)
            self.F = nn.Parameter(U_b[:, :n_comp] @ torch.diag(torch.sqrt(torch.abs(S_b[:n_comp]) + 1e-6)))
            self.G = nn.Parameter(U_w[:, :n_comp] @ torch.diag(torch.sqrt(torch.abs(S_w[:n_comp]) + 1e-6)))
            self.Sigma = nn.Parameter(torch.ones(D, device=device))
        
        self.mean = nn.Parameter(mean.squeeze(0))
        
        for iteration in range(self.n_iter):
            self._em_step(x, self.train_labels)
        
        self._is_fitted = True

    def _em_step(self, x, labels):
        N, D = x.shape
        device = x.device
        cov_total = (self.F @ self.F.t()) + (self.G @ self.G.t()) + torch.diag(self.Sigma)
        precision_total = torch.linalg.inv(cov_total + self.reg * torch.eye(D, device=device))
        new_sigma = torch.diag(torch.linalg.inv(precision_total + self.reg * torch.eye(D, device=device)))
        self.Sigma = nn.Parameter(torch.clamp(new_sigma, min=1e-6))
