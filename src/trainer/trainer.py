import contextlib
import numpy as np
import torch
from src.logger.utils import (
    feature_plot_params_from_config,
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
        """
        Context manager for automatic mixed precision training.
        
        Returns:
            context manager: torch.autocast context or nullcontext
        """
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
        metrics: MetricTracker|None=None,
        backend=None
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
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)
            if self.is_train:
                all_losses=self.criterion(**batch)
                batch.update(all_losses)
        
        if self.is_train:
            # Get accumulation steps from config
            accumulation_steps = self.config.trainer.get("accumulation_steps", 1)
            
            # Zero gradients at the start of each accumulation cycle
            if self._batch_count % accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Scale loss for gradient accumulation
            scaled_loss = batch["loss"] / accumulation_steps
            scaled_loss.backward()
            
            # Update batch counter
            self._batch_count += 1
                
            # Only update weights on accumulation step
            if self._batch_count % accumulation_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and self._scheduler_steps_each_batch():
                    self.lr_scheduler.step()
        
        if backend is not None:
            batch["pred"]=backend.predict(batch["embedding"].detach().cpu())
        if metrics is not None:
            metrics.update(batch)
        return batch

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
                sample_rate=int(batch.get("sample_rate")[0]),
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
