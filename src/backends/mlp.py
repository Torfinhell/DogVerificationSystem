import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
from src.backends.base import BaseBackend

logger = logging.getLogger(__name__)


class MLPBackend(BaseBackend):
    NAME = "MLP"

    def __init__(
        self,
        labels,
        hidden_dim=512,
        n_layers=2,
        dropout=0.0,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        weight_decay=0.0,
        normalize_inputs=True,
    ):
        super().__init__()
        self.register_buffer(
            "label_reference",
            torch.tensor(labels, dtype=torch.long)
            if not isinstance(labels, torch.Tensor)
            else labels.clone().detach().long(),
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.normalize_inputs = normalize_inputs

        self.classifier = None
        self._is_fitted = False
        self.label_to_index = None

    def reset(self):
        self.classifier = None
        self._is_fitted = False
        self.label_to_index = None

    def _build_classifier(self, input_dim, output_dim):
        layers = []
        hidden_dim = self.hidden_dim
        for i in range(self.n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < self.n_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                layers.append(nn.ReLU())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def _prepare_data(self, embeddings, labels):
        device = embeddings.device
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        if self.normalize_inputs:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        self.label_to_index = {
            int(label.item()): idx
            for idx, label in enumerate(self.label_reference)
        }
        mapped_labels = torch.tensor(
            [self.label_to_index[int(label.item())] for label in labels],
            device=device,
            dtype=torch.long,
        )
        return embeddings, mapped_labels

    def fit(self, embeddings, labels):
        self.reset()

        if labels is None:
            raise ValueError("MLPBackend requires labels for fitting")

        device = embeddings.device
        embeddings, mapped_labels = self._prepare_data(embeddings, labels)
        num_classes = len(self.label_reference)
        input_dim = embeddings.shape[1]

        self.classifier = self._build_classifier(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(embeddings, mapped_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.classifier.train()
        for _ in range(self.epochs):
            for batch_embeddings, batch_labels in dataloader:
                optimizer.zero_grad()
                logits = self.classifier(batch_embeddings)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

        self._is_fitted = True

    def predict(self, embeddings):
        if not self._is_fitted or self.classifier is None:
            raise ValueError("Backend not fitted")

        device = next(self.classifier.parameters()).device
        embeddings = embeddings.to(device)
        if self.normalize_inputs:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        logits = self.classifier(embeddings)
        indices = logits.argmax(dim=-1)
        return self.label_reference[indices]
