# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import avex

# class AVEXModel(nn.Module):
#     """Wrapper for AVEX pretrained models and feature extractors."""

#     def __init__(
#         self,
#         model_name: str = "esp_aves2_sl_beats_all",
#         device: str = "cpu",
#         return_features_only: bool = True,
#         freeze_backbone: bool = True,
#         num_classes: int | None = None,
#         probe_hidden_dim: int = 768,
#         probe_dropout: float = 0.0,
#         normalize_features: bool = True,
#     ):
#         super().__init__()
#         self.device = device
#         self.model_name = model_name
#         self.return_features_only = return_features_only
#         self.freeze_backbone = freeze_backbone
#         self.num_classes = num_classes
#         self.normalize_features = normalize_features

#         self.backbone = avex.load_model(
#             self.model_name,
#             return_features_only=self.return_features_only,
#             device=self.device,
#         )

#         if self.freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         self.embedding_dim = self._infer_embedding_dim()
#         self.probe = None
#         if self.num_classes is not None:
#             self.probe = nn.Sequential(
#                 nn.Linear(self.embedding_dim, probe_hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(probe_dropout),
#                 nn.Linear(probe_hidden_dim, self.num_classes),
#             )

#     def _infer_embedding_dim(self):
#         dummy = torch.zeros(1, 16000, device=self.device)
#         with torch.no_grad():
#             output = self.backbone(dummy)

#         if isinstance(output, dict):
#             features = output.get("features") or output.get("embedding") or output.get("logits")
#         elif isinstance(output, tuple):
#             features = output[0]
#         else:
#             features = output

#         if features.ndim == 3:
#             return features.size(-1)
#         return features.size(-1)

#     def forward(self, audio, **batch):
#         if audio.ndim == 1:
#             audio = audio.unsqueeze(0)
#         if audio.ndim == 2 and audio.size(0) == 1 and audio.size(1) == 0:
#             raise ValueError("Audio tensor must contain samples")

#         audio = audio.to(self.device)
#         with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
#             output = self.backbone(audio)

#         if isinstance(output, dict):
#             features = output.get("features") or output.get("embedding") or output.get("logits")
#         elif isinstance(output, tuple):
#             features = output[0]
#         else:
#             features = output

#         if features.ndim == 3:
#             embeddings = features.mean(dim=1)
#         else:
#             embeddings = features

#         if self.normalize_features:
#             embeddings = F.normalize(embeddings, dim=1)

#         if self.probe is not None:
#             logits = self.probe(embeddings)
#             return {"logits": logits, "embedding": embeddings}

#         return {"logits": embeddings, "embedding": embeddings}
