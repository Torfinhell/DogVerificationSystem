from abc import abstractmethod
import contextlib

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.metrics.epoch_metric import EpochMetric
from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
from src.utils.hydra_cfg import cfg_get
from src.utils.plot_utils import embedding_to_3d

class BaseTrainer:
    """Input: train config/runtime deps. Output: train/eval orchestration."""

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
        backends=None,
    ):
        """Input: model stack, loaders, metrics. Output: initialized trainer state."""
        self.is_train = True
        self.current_eval_part = None

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.backends = backends or []

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0  
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = (
            self.cfg_trainer.save_period
        )
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = metrics
        self.metric_keys = self.metrics.get("metric_keys", {})
        self.backend_keys = self.metrics.get("backend_keys", [])
        self.backend_metric_objects = self.metrics.get("test", {}).get("backends", {})
        _train_metric_keys = (
            *self.config.writer.logger.loss_names,
            "grad_norm",
            *self.metric_keys.get("train", []),
        )
        self.train_metrics = MetricTracker(
            *_train_metric_keys,
            writer=self.writer,
        )
        self.epoch_train_metrics = MetricTracker(*_train_metric_keys, writer=None)

        self.val_metric_objects = self.metrics.get("val", [])
        self.val_metrics = MetricTracker(
            *self.metric_keys.get("val", []),
            writer=self.writer,
        )
        self.test_metrics = {}
        for backend_name in self.backend_keys:
            self.test_metrics[backend_name] = MetricTracker(
                *self.metric_keys.get("test_backends", {}).get(backend_name, []),
                writer=self.writer,
            )
        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.logger.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def _autocast(self):
        """Input: none. Output: autocast context manager."""
        return contextlib.nullcontext()

    def _collect_embeddings(self, dataloader, max_samples: int = None, max_batches: int = None):
        """Input: dataloader. Output: embeddings tensor and labels tensor."""
        embs, labs = [], []
        n = 0
        self.model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(dataloader):
                if max_batches is not None and bi >= max_batches:
                    break
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)
                with self._autocast():
                    out = self.model(**batch)
                e = out.get("embedding")
                if e is None:
                    e = out["logits"]
                emb = e.detach().cpu()
                y = batch["label"].detach().cpu()
                embs.append(emb)
                labs.append(y)
                n += emb.shape[0]
                if max_samples is not None and n >= max_samples:
                    break
        if not embs:
            return None, None
        embeddings = torch.cat(embs, dim=0)
        labels = torch.cat(labs, dim=0)
        if max_samples is not None:
            embeddings = embeddings[:max_samples]
            labels = labels[:max_samples]
        return embeddings, labels



    def _wandb_after_train_epoch(self, epoch, logs):
        if self.writer is None:
            return
        if getattr(self.writer, "wandb", None) is None:
            return
        if hasattr(self.writer, "log_epoch_summary"):
            self.writer.set_step(epoch * self.epoch_len, "epoch_end")
            self.writer.log_epoch_summary(logs, epoch)
        wcfg = self.config.get("writer")
        if wcfg is None or not cfg_get(wcfg, "plot_3d", False):
            return
        max_s = cfg_get(wcfg, "embedding_max_samples", 2048)
        max_b = cfg_get(wcfg, "embedding_max_batches", 64)
        self.model.eval()
        for part, dl in self.evaluation_dataloaders.items():
            self.writer.set_step(epoch * self.epoch_len, part)
            embs, labs = self._collect_embeddings(dl, max_s, max_b)
            if embs is not None:
                pts = embedding_to_3d(embs)
                self.writer.add_plot_3d(
                    "embedding_3d", pts.numpy(), labs.numpy(), title=f"{part} PCA epoch {epoch}"
                )

    def update_train_dataloader(self):
        """Input: current criterion weights. Output: refreshed train dataloader."""
        similarity_matrix = self.criterion.embedding.weight @ self.criterion.embedding.weight.T
        batch_sampler = instantiate(
            self.config.batch_sampler.sampler,
            ds=self.train_dataloader.dataset,
            similarity_matrix=similarity_matrix
        )

        new_dataloader = instantiate(
            self.config.dataloader.dataloader_with_batch_sampler,
            dataset=self.train_dataloader.dataset,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            batch_sampler=batch_sampler,
        )

        if self.cfg_trainer.get("epoch_len", None) is None:
            self.train_dataloader = new_dataloader
        else:
            self.train_dataloader = inf_loop(new_dataloader)

    def train(self):
        """Input: none. Output: executed training loop."""
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """Input: none. Output: epoch-wise train/eval with checkpointing."""
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            self._wandb_after_train_epoch(epoch, result)
            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        """Input: epoch index. Output: train and eval logs for epoch."""
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.epoch_train_metrics.reset()
        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.epoch_len)
            self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    epoch_metrics=self.epoch_train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())
            self.epoch_train_metrics.update("grad_norm", self._get_grad_norm())
            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_scalars(self.train_metrics)
                    self._log_batch(batch_idx, batch)
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = self.epoch_train_metrics.result()

        for part, dataloader in self.evaluation_dataloaders.items():
            part_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in part_logs.items()})
        if self.config.get("batch_sampler", None) is not None:
            self.update_train_dataloader()
        self._reset_stateful_metrics()
        return logs

    def _reset_stateful_metrics(self):
        """Input: metric objects. Output: metrics reset for next epoch."""
        metric_lists = [self.metrics.get("train", []), self.metrics.get("val", [])]
        metric_lists.extend(self.backend_metric_objects.values())
        for metric_list in metric_lists:
            for met in metric_list:
                if hasattr(met, "metric") and hasattr(met.metric, "reset"):
                    met.metric.reset()

    def _evaluation_epoch(self, epoch, part, dataloader):
        """Input: epoch, partition, dataloader. Output: partition evaluation logs."""
        self.is_train = False
        self.model.eval()
        self.current_eval_part = part
        metrics_for_part = self.metrics.get(part, [])
        if not isinstance(metrics_for_part, list):
            metrics_for_part = []
        batch_metrics = [m for m in metrics_for_part if not isinstance(m, EpochMetric)]
        epoch_metrics = [m for m in metrics_for_part if isinstance(m, EpochMetric)]

        logs = {}

        if part == "val" and self.backends:
            all_embeddings = []
            all_labels = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = self.move_batch_to_device(batch)
                    batch = self.transform_batch(batch)
                    with self._autocast():
                        out = self.model(**batch)
                        e = out.get("embedding")
                        if e is None:
                            e = out["logits"]
                        emb = e.detach().cpu()
                        y = batch["label"].detach().cpu()
                        all_embeddings.append(emb)
                        all_labels.append(y)
            if all_embeddings:
                all_embeddings_cat = torch.cat(all_embeddings, dim=0).to(self.device)
                all_labels_cat = torch.cat(all_labels, dim=0)
                for backend in self.backends:
                    try:
                        backend.fit(all_embeddings_cat, labels=all_labels_cat)
                        self.logger.info(f"Backend {backend.__class__.__name__} fitted on {all_embeddings_cat.shape[0]} embeddings")
                    except Exception as e:
                        self.logger.error(f"Error fitting backend: {e}")

        if part == "val":
            self.val_metrics.reset()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"{part} main", total=len(dataloader)):
                    batch = self.process_batch(batch, metrics=self.val_metrics)
                    if "logits" in batch:
                        all_predictions.append(batch["logits"].detach().cpu())
                    elif "preds" in batch:
                        all_predictions.append(batch["preds"].detach().cpu())
                    if "label" in batch:
                        all_labels.append(batch["label"].detach().cpu())
                    if self.writer is not None and batch_idx == 0:
                        self.writer.set_step(epoch * self.epoch_len, part)
                        self._log_batch(batch_idx, batch, part)
            if self.writer is not None:
                self.writer.set_step(epoch * self.epoch_len, part)
                self._log_scalars(self.val_metrics)
            results = self.val_metrics.result()
            logs.update(**{f"{name}": value for name, value in results.items()})
            for met in epoch_metrics:
                extra = met.finalize()
                logs.update(**{f"{k}": v for k, v in extra.items()})
                if self.writer is not None:
                    self.writer.set_step(epoch * self.epoch_len, part)
                    for k, v in extra.items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            v = float(v.detach().cpu().item())
                        if isinstance(v, (float, int)) and not (isinstance(v, float) and v != v):
                            self.writer.add_scalar(k, v)
                met.reset()

            if all_predictions and all_labels:
                all_predictions_cat = torch.cat(all_predictions, dim=0)
                all_labels_cat = torch.cat(all_labels, dim=0)
                if self.writer is not None and getattr(self.writer, "wandb", None) is not None and cfg_get(self.config, "writer.log_confusion_matrix_image", True) and hasattr(self.writer, "add_confusion_matrix_image"):
                    self.writer.set_step(epoch * self.epoch_len, part)
                    self.writer.add_confusion_matrix_image("confusion_matrix", preds=all_predictions_cat, labels=all_labels_cat, title=f"{part} epoch {epoch}")

        for backend_name, backend in zip(self.backend_keys, self.backends):
            if hasattr(backend, '_is_fitted') and backend._is_fitted:
                backend_metrics = self.test_metrics.get(backend_name)
                backend_metrics.reset()
                backend_metric_objects = self.backend_metric_objects.get(backend_name, [])
                all_backend_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"{part} backend {backend_name}", total=len(dataloader)):
                        batch = self.move_batch_to_device(batch)
                        batch = self.transform_batch(batch)
                        with self._autocast():
                            out = self.model(**batch)
                            batch.update(out)
                            if "embedding" in batch:
                                embeddings = batch["embedding"]
                                preds = backend.predict(embeddings)
                                batch["preds"] = preds
                                all_backend_preds.append(preds.detach().cpu())
                            if "label" in batch:
                                all_labels.append(batch["label"].detach().cpu())
                            if not self.is_train and "AAMSoftmax" in self.config.loss_function._target_:
                                batch["loss"] = torch.tensor(0.0, device=self.device)
                            else:
                                all_losses = self.criterion(**batch)
                                batch.update(all_losses)
                        for met in backend_metric_objects:
                            if met.name == "confusion_matrix" or isinstance(met, EpochMetric):
                                met(**batch)
                            else:
                                met_value = met(**batch)
                                if isinstance(met_value, torch.Tensor):
                                    if met_value.numel() == 1:
                                        met_value = float(met_value.detach().cpu().item())
                                    else:
                                        continue
                                if isinstance(met_value, (float, int)) and not (isinstance(met_value, float) and (met_value != met_value)):
                                    backend_metrics.update(met.name, met_value)
                        if self.writer is not None and batch_idx == 0:
                            self.writer.set_step(epoch * self.epoch_len, part)
                            self._log_batch(batch_idx, batch, part)
                if self.writer is not None:
                    self.writer.set_step(epoch * self.epoch_len, part)
                    self._log_scalars(backend_metrics)
                backend_results = backend_metrics.result()
                logs.update(**{f"{backend_name}_{name}": value for name, value in backend_results.items()})
                for met in backend_metric_objects:
                    if isinstance(met, EpochMetric):
                        extra = met.finalize()
                        logs.update(**{f"{backend_name}_{k}": v for k, v in extra.items()})
                        if self.writer is not None:
                            self.writer.set_step(epoch * self.epoch_len, part)
                            for k, v in extra.items():
                                if isinstance(v, torch.Tensor) and v.numel() == 1:
                                    v = float(v.detach().cpu().item())
                                if isinstance(v, (float, int)) and not (isinstance(v, float) and v != v):
                                    self.writer.add_scalar(f"{backend_name}_{k}", v)
                        met.reset()
                if all_backend_preds and all_labels:
                    all_backend_preds_cat = torch.cat(all_backend_preds, dim=0)
                    all_labels_cat = torch.cat(all_labels, dim=0)
                    if self.writer is not None and getattr(self.writer, "wandb", None) is not None and cfg_get(self.config, "writer.log_confusion_matrix_image", True) and hasattr(self.writer, "add_confusion_matrix_image"):
                        self.writer.set_step(epoch * self.epoch_len, part)
                        self.writer.add_confusion_matrix_image(
                            f"confusion_matrix_{backend_name}",
                            preds=all_backend_preds_cat,
                            labels=all_labels_cat,
                            title=f"{part} backend {backend_name} epoch {epoch}",
                        )

        self.current_eval_part = None
        return logs

    def _monitor_performance(self, logs, not_improved_count):
        """Input: logs and patience state. Output: best flag, stop flag, counter."""
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """Input: batch dict. Output: batch with tensors moved to device."""
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """Input: batch dict. Output: batch after optional batch transforms."""
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """Input: gradients. Output: gradients clipped by configured max norm."""
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """Input: norm type. Output: scalar gradient norm."""
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """Input: batch index. Output: formatted progress string."""
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """Input: batch index/data and mode. Output: logged batch artifacts."""
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """Input: metric tracker. Output: scalar metrics logged to writer."""
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """Input: epoch and save flags. Output: checkpoint files written."""
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.logger.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.logger.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """Input: checkpoint path. Output: restored model/optimizer/scheduler."""
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        csd = checkpoint.get("criterion_state_dict")
        if csd is not None:
            self.criterion.load_state_dict(csd)

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """Input: pretrained checkpoint path. Output: loaded model weights."""
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
