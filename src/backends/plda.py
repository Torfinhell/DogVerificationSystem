"""PLDA (Probabilistic Linear Discriminant Analysis) backend for speaker/dog verification."""

import torch
import torch.nn as nn
import logging

from src.backends.base import BaseBackend

logger = logging.getLogger(__name__)


class PLDABackend(BaseBackend):
    """
    PLDA (Probabilistic Linear Discriminant Analysis) backend.
    
    Implements PLDA-based similarity scoring for speaker/dog verification.
    PLDA is a generative model that learns to separate within-speaker and
    between-speaker variability in embedding space.
    
    Training uses EM algorithm to estimate:
    - Global mean
    - Between-speaker subspace F
    - Within-speaker subspace G
    - Residual covariance Sigma
    
    Scoring uses log-likelihood ratio between same-speaker and different-speaker hypotheses.
    """

    def __init__(self, n_components=256, n_iter=10, reg=1e-6, **kwargs):
        """
        Initialize the PLDA backend.
        
        Args:
            n_components (int): Number of latent factors in PLDA model.
                Default: 256
            n_iter (int): Number of iterations for EM training.
                Default: 10
            reg (float): Regularization for covariance matrices.
                Default: 1e-6
            **kwargs: Additional arguments for PLDA configuration
        """
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.reg = reg
        self.config = kwargs
        
        # PLDA parameters
        self.register_buffer("mean", None)  
        self.register_buffer("F", None)     
        self.register_buffer("G", None)     
        self.register_buffer("Sigma", None) 
        self._is_fitted = False        

    def forward(self, embeddings1, embeddings2):
        """
        Forward pass - compute PLDA similarity scores.
        
        Args:
            embeddings1: Embedding matrix of shape (N, D)
            embeddings2: Embedding matrix of shape (M, D)
            
        Returns:
            scores: PLDA scores of shape (N, M)
        """
        if not self._is_fitted:
            logger.warning("PLDA backend not fitted yet. Computing cosine similarity as fallback.")
            return torch.nn.functional.normalize(embeddings1, p=2, dim=1) @ torch.nn.functional.normalize(embeddings2, p=2, dim=1).t()
        N, D = embeddings1.shape
        M, _ = embeddings2.shape
        device = embeddings1.device
        mean = self.mean.to(device)
        F = self.F.to(device)
        G = self.G.to(device)
        Sigma = self.Sigma.to(device)
        x1 = embeddings1 - mean  
        x2 = embeddings2 - mean  
        FF = F @ F.t()  
        GG = G @ G.t()  
        Sigma_diag = torch.diag(Sigma)
        cov_total = FF + GG + Sigma_diag  
        cov_within = GG + Sigma_diag  
        precision_total = torch.linalg.inv(cov_total + self.reg * torch.eye(D, device=device))
        precision_within = torch.linalg.inv(cov_within + self.reg * torch.eye(D, device=device))
        _, logdet_total = torch.linalg.slogdet(cov_total + self.reg * torch.eye(D, device=device))
        _, logdet_within = torch.linalg.slogdet(cov_within + self.reg * torch.eye(D, device=device))
        scores = torch.zeros(N, M, device=device)
        for i in range(N):
            for j in range(M):
                xi = x1[i:i+1]  
                xj = x2[j:j+1]  
                x_concat = torch.cat([xi, xj], dim=0)  
                sum_x = xi + xj 
                term1 = (sum_x @ precision_total @ sum_x.t()).item() / 4.0
                term2 = (xi @ precision_total @ xi.t()).item() / 2.0
                term3 = (xj @ precision_total @ xj.t()).item() / 2.0
                term4 = (logdet_total - 2 * logdet_within) / 4.0
                
                score = term1 - term2 - term3 + term4
                scores[i, j] = score
        
        return scores

    def fit(self, embeddings, labels=None):
        N, D = embeddings.shape
        device = embeddings.device
        
        logger.info(f"Fitting PLDA on {N} embeddings of dimension {D}")
        
        # Center embeddings
        mean = embeddings.mean(dim=0, keepdim=True)  # (1, D)
        x = embeddings - mean  # (N, D)
        if labels is None:
            cov = (x.t() @ x) / (N - 1)
            U, S, _ = torch.linalg.svd(cov)
            self.F = nn.Parameter(U[:, :min(self.n_components, D)] @ torch.diag(torch.sqrt(S[:min(self.n_components, D)])))
            self.G = nn.Parameter(U[:, :min(self.n_components, D)] @ torch.diag(torch.sqrt(S[:min(self.n_components, D)]) * 0.1))
            self.Sigma = nn.Parameter(torch.ones(D, device=device))
        else:
            unique_labels = torch.unique(labels)
            n_speakers = len(unique_labels)
            
            # Between-speaker 
            speaker_means = []
            for label in unique_labels:
                mask = labels == label
                speaker_mean = x[mask].mean(dim=0)
                speaker_means.append(speaker_mean)
            speaker_means = torch.stack(speaker_means)  
            
            cov_between = (speaker_means.t() @ speaker_means) / n_speakers
            
            # Within-speaker 
            cov_within = 0
            for label in unique_labels:
                mask = labels == label
                x_speaker = x[mask]
                if len(x_speaker) > 1:
                    cov_within = cov_within + (x_speaker.t() @ x_speaker) / (len(x_speaker) - 1)
            cov_within = cov_within / n_speakers
            
            # SVD 
            U_b, S_b, _ = torch.linalg.svd(cov_between)
            U_w, S_w, _ = torch.linalg.svd(cov_within)
            
            n_comp = min(self.n_components, D)
            self.F = nn.Parameter(U_b[:, :n_comp] @ torch.diag(torch.sqrt(torch.abs(S_b[:n_comp]) + 1e-6)))
            self.G = nn.Parameter(U_w[:, :n_comp] @ torch.diag(torch.sqrt(torch.abs(S_w[:n_comp]) + 1e-6)))
            self.Sigma = nn.Parameter(torch.ones(D, device=device))
        
        self.mean = nn.Parameter(mean.squeeze(0))
        
        # EM
        for iteration in range(self.n_iter):
            logger.info(f"PLDA EM iteration {iteration + 1}/{self.n_iter}")
            self._em_step(x, labels)
        
        self._is_fitted = True
        logger.info(f"PLDA training complete")

    def _em_step(self, x, labels):
        """Single EM step for PLDA parameter estimation."""
        N, D = x.shape
        device = x.device
        
        F = self.F
        G = self.G
        Sigma = self.Sigma
        
        # E-step
        FF = F @ F.t()  
        GG = G @ G.t() 
        Sigma_diag = torch.diag(Sigma)
        cov_total = FF + GG + Sigma_diag
        # cov_within = GG + Sigma_diag
        precision_total = torch.linalg.inv(cov_total + self.reg * torch.eye(D, device=device))
        # precision_within = torch.linalg.inv(cov_within + self.reg * torch.eye(D, device=device))
        total_var = 0
        count = 0
        for i in range(N):
            xi = x[i:i+1]
            residual = xi @ precision_total @ xi.t()
            total_var = total_var + residual.item()
            count += 1
        new_sigma = torch.sqrt(torch.clamp(Sigma, min=1e-6))
        self.Sigma = nn.Parameter(new_sigma)


