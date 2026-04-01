import contextlib

import numpy as np
import torch
from src.metrics.epoch_metric import EpochMetric
from src.utils.hydra_cfg import cfg_get
from src.utils.plot_utils import (
    feature_plot_params_from_config,
    plot_images,
    plot_mfcc_coeffs,
    plot_spectrogram,
)
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.torch_utils import str_to_dtype
import torch.nn.functional as F



class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def _autocast(self):
        amp = self.config.trainer.get("amp") or {}
        if not amp.get("enabled"):
            return contextlib.nullcontext()
        if torch.device(self.device).type != "cuda":
            return contextlib.nullcontext()
        dtype = str_to_dtype(amp.get("dtype", "bfloat16"))
        return torch.autocast(device_type="cuda", dtype=dtype)

    def process_batch(
        self,
        batch,
        metrics: MetricTracker,
        epoch_metrics: MetricTracker | None = None,
    ):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
            epoch_metrics (MetricTracker | None): optional second tracker updated
                in parallel for full-epoch averages (train only).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)
        if not self.is_train and "embedding" in batch:
            embeddings = batch["embedding"]
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            cos_scores = torch.mm(embeddings_norm, embeddings_norm.t())
            batch["cos_scores"] = cos_scores
            if self.backend is not None:
                batch_scores = self.backend(embeddings, embeddings)
                batch["backend_scores"] = batch_scores

        if self.is_train:
            batch["loss"].backward() 
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.logger.loss_names:
            v = batch[loss_name].item()
            metrics.update(loss_name, v)
            if epoch_metrics is not None:
                epoch_metrics.update(loss_name, v)

        for met in metric_funcs:
            if met.name == "confusion_matrix" or isinstance(met, EpochMetric):
                met(**batch)
                continue

            met_value = met(**batch)
            
            if isinstance(met_value, torch.Tensor):
                if met_value.numel() == 1:
                    met_value = float(met_value.detach().cpu().item())
                else:
                    continue

            if isinstance(met_value, (float, int)) and not (isinstance(met_value, float) and (met_value != met_value)):  # Check for NaN
                metrics.update(met.name, met_value)
                if epoch_metrics is not None:
                    epoch_metrics.update(met.name, met_value)
            elif met_value is not None:
                try:
                    fv = float(met_value)
                    metrics.update(met.name, fv)
                    if epoch_metrics is not None:
                        epoch_metrics.update(met.name, fv)
                except Exception:
                    pass

        return batch

    @staticmethod
    def _log_batch_sample_rate(batch: dict) -> int | None:
        sr = batch.get("sample_rate")
        if sr is None:
            return None
        if isinstance(sr, (list, tuple)):
            return int(sr[0])
        if isinstance(sr, torch.Tensor):
            return int(sr[0].item())
        return int(sr)

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Train: called every ``log_step`` (shuffled order each epoch).
        Val: called once per partition per epoch (first batch; order is fixed
        when the dataloader does not shuffle).

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): ``train`` or eval partition name.
        """
        if self.writer is None:
            return
        if getattr(self.writer, "wandb", None) is None:
            return

        if "audio" in batch:
            audio_sample = batch["audio"][0]
            if "audio_lengths" in batch:
                original_length = batch["audio_lengths"][0].item()
                audio_sample = audio_sample[:original_length]
            
            self.writer.add_audio(
                "audio_sample",
                audio_sample,
                sample_rate=self._log_batch_sample_rate(batch),
            )

        if "spectral_feat" in batch:
            feat = batch["spectral_feat"][0]
            if "spectral_feat_lengths" in batch:
                original_length = batch["spectral_feat_lengths"][0].item()
                if feat.dim() == 2 and feat.shape[1] > original_length:
                    feat = feat[:, :original_length]
            
            params = feature_plot_params_from_config(self.config)
            if params.get("kind") == "mfcc":
                img = plot_mfcc_coeffs(feat, params, title="MFCC")
                self.writer.add_image("mfcc", img)
            else:
                img = plot_spectrogram(
                    feat,
                    mel_spec={
                        "hop_length": params["hop_length"],
                        "n_mels": params["n_mels"],
                        "sample_rate": params["sample_rate"],
                    },
                    title="Mel spectrogram",
                )
                self.writer.add_image("spectrogram", img)

        if "img" in batch:
            imgs = batch["img"]
            names = cfg_get(self.config, "writer.log_batch_image_names", []) or []
            figsize = cfg_get(self.config, "writer.log_batch_figsize", (12, 4))
            if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
                figsize = (float(figsize[0]), float(figsize[1]))
            else:
                figsize = (12.0, 4.0)
            b = min(len(names), imgs.shape[0]) if isinstance(names, (list, tuple)) else 0
            if b >= 2:
                panel = plot_images(
                    imgs[:b], [str(x) for x in names[:b]], figsize=figsize
                )
                self.writer.add_image("images", panel)
            else:
                im = imgs[0].detach().cpu().float().numpy()
                if im.ndim == 3:
                    im = np.transpose(im, (1, 2, 0))
                elif im.ndim == 2:
                    im = np.stack([im, im, im], axis=-1)
                self.writer.add_image("image", im)
