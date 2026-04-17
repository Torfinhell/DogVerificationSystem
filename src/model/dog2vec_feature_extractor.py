# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import fairseq

# class FairseqFeatureExtractor(nn.Module):
#     """Wrapper for Fairseq checkpoint-based feature extraction."""

#     def __init__(
#         self,
#         checkpoint_path: str,
#         device: str = "cuda",
#         layer: int = 9,
#         max_chunk: int = 1600000,
#         num_classes: int | None = None,
#         freeze_backbone: bool = True,
#         pooling: str = "mean",
#     ):
#         super().__init__()

#         self.checkpoint_path = checkpoint_path
#         self.device = torch.device(device)
#         self.layer = layer
#         self.max_chunk = max_chunk
#         self.num_classes = num_classes
#         self.freeze_backbone = freeze_backbone
#         self.pooling = pooling

#         models, task, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([
#             self.checkpoint_path
#         ])
#         self.task = task
#         self.backbone = models[0].eval().to(self.device)

#         if self.freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         self.embedding_dim = self._infer_embedding_dim()
#         self.probe = None
#         if self.num_classes is not None:
#             self.probe = nn.Linear(self.embedding_dim, self.num_classes)

#     def _infer_embedding_dim(self):
#         dummy = torch.zeros(1, 16000, device=self.device)
#         if self.task.cfg.normalize:
#             dummy = F.layer_norm(dummy, dummy.shape)

#         with torch.no_grad():
#             feat, _ = self.backbone.extract_features(
#                 source=dummy,
#                 padding_mask=None,
#                 mask=False,
#                 output_layer=self.layer,
#             )
#         return feat.size(-1)

#     def _pool(self, features):
#         if self.pooling == "mean":
#             return features.mean(dim=1)
#         if self.pooling == "max":
#             return features.max(dim=1).values
#         return features.mean(dim=1)

#     def forward(self, audio, **batch):
#         if audio.ndim == 1:
#             audio = audio.unsqueeze(0)
#         audio = audio.to(self.device)
#         if self.task.cfg.normalize:
#             audio = F.layer_norm(audio, audio.shape)

#         features_chunks = []
#         for start in range(0, audio.size(1), self.max_chunk):
#             chunk = audio[:, start : start + self.max_chunk]
#             feat_chunk, _ = self.backbone.extract_features(
#                 source=chunk,
#                 padding_mask=None,
#                 mask=False,
#                 output_layer=self.layer,
#             )
#             features_chunks.append(feat_chunk)

#         features = torch.cat(features_chunks, dim=1)
#         embeddings = self._pool(features)

#         if self.probe is not None:
#             logits = self.probe(embeddings)
#             return {"logits": logits, "embedding": embeddings}

#         return {"logits": embeddings, "embedding": embeddings}
