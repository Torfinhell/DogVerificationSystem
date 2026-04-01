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
    """
    Base class for all trainers.
    """

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
        backend=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            backend (nn.Module | None): backend for computing similarity scores
                during evaluation/inference (e.g., CosineBackend, PLDABackend).
        """
        self.is_train = True

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
        self.backend = backend

        
        self.train_dataloader_raw = dataloaders["train"]
        self.train_dataloader = self.train_dataloader_raw
        if epoch_len is None:
            
            self.epoch_len = len(self.train_dataloader_raw)
        else:
            
            self.train_dataloader = inf_loop(self.train_dataloader_raw)
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
        _train_metric_keys = (
            *self.config.writer.logger.loss_names,
            "grad_norm",
            *[
                m.name
                for m in self.metrics["train"]
                if not isinstance(m, EpochMetric)
            ],
        )
        self.train_metrics = MetricTracker(
            *_train_metric_keys,
            writer=self.writer,
        )
        self.epoch_train_metrics = MetricTracker(*_train_metric_keys, writer=None)
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.logger.loss_names,
            *[
                m.name
                for m in self.metrics["inference"]
                if not isinstance(m, EpochMetric)
            ],
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
        """Trainer overrides with AMP autocast when enabled."""
        return contextlib.nullcontext()

    def _gather_embeddings(self, dataloader, max_samples: int, max_batches: int):
        embs, labs = [], []
        n = 0
        self.model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(dataloader):
                if bi >= max_batches:
                    break
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)
                with self._autocast():
                    out = self.model(**batch)
                e = out.get("embedding")
                if e is None:
                    e = out["logits"]
                emb = e.detach().cpu()
                y = batch["labels"].detach().cpu()
                embs.append(emb)
                labs.append(y)
                n += emb.shape[0]
                if n >= max_samples:
                    break
        if not embs:
            return None, None
        emb = torch.cat(embs, dim=0)[:max_samples]
        lab = torch.cat(labs, dim=0)[:max_samples]
        pts = embedding_to_3d(emb)
        return pts.numpy(), lab.numpy()

    def _fit_backend_after_epoch(self):
        """
        Fit the backend (e.g., PLDA) on all training embeddings after an epoch.
        
        This method should be called after each training epoch to update the
        backend model with the latest embeddings from the training set.
        """
        if self.backend is None or not hasattr(self.backend, "fit"):
            return
        
        # Collect all embeddings from training dataloader
        embeddings_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                self.train_dataloader_raw,
                desc="Collecting embeddings for backend",
                total=len(self.train_dataloader_raw),
            ):
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)
                with self._autocast():
                    out = self.model(**batch)
                
                # Extract embeddings
                emb = out.get("embedding")
                if emb is None:
                    emb = out.get("logits")
                
                if emb is not None:
                    embeddings_list.append(emb.detach().cpu())
                    labels_list.append(batch["labels"].detach().cpu())
        
        if not embeddings_list:
            self.logger.warning("No embeddings collected for backend fitting")
            return
        
        # Concatenate all embeddings and labels
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0) if labels_list else None
        
        # Fit the backend
        try:
            self.backend.fit(all_embeddings.to(self.device), labels=all_labels)
            self.logger.info(f"Backend {self.backend.__class__.__name__} fitted on {all_embeddings.shape[0]} embeddings")
        except Exception as e:
            self.logger.error(f"Error fitting backend: {e}")

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
        self.writer.set_step(epoch * self.epoch_len, "train")
        e, y = self._gather_embeddings(self.train_dataloader_raw, max_s, max_b)
        if e is not None:
            self.writer.add_plot_3d(
                "embedding_3d", e, y, title=f"train PCA epoch {epoch}"
            )
        for part, dl in self.evaluation_dataloaders.items():
            self.writer.set_step(epoch * self.epoch_len, part)
            e, y = self._gather_embeddings(dl, max_s, max_b)
            if e is not None:
                self.writer.add_plot_3d(
                    "embedding_3d", e, y, title=f"{part} PCA epoch {epoch}"
                )

    def update_train_dataloader(self):
        similarity_matrix=(self.criterion.embedding.weight@self.criterion.embedding.weight.T)
        batch_sampler = instantiate(
            self.config.batch_sampler.sampler,
            ds=self.train_dataloader_raw.dataset,
            similarity_matrix=similarity_matrix
        )

        new_dataloader = instantiate(
            self.config.dataloader.dataloader_with_batch_sampler,
            dataset=self.train_dataloader_raw.dataset,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            batch_sampler=batch_sampler,
        )

        self.train_dataloader_raw = new_dataloader
        if self.cfg_trainer.get("epoch_len", None) is None:
            self.train_dataloader = new_dataloader
        else:
            self.train_dataloader = inf_loop(new_dataloader)

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            
            # Fit backend (e.g., PLDA) on all training embeddings after epoch
            self._fit_backend_after_epoch()
            
            # logging embeddings (PCA 3D) + W&B epoch_summary scalars
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
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
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
            val_logs = self._evaluation_epoch(epoch, part, dataloader)

            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items() if name != "confusion_matrix"})

        if self.config.get("batch_sampler", None) is not None:
            self.update_train_dataloader()
        self._reset_stateful_metrics()

        return logs

    def _reset_stateful_metrics(self):
        """
        Reset stateful torchmetrics at the end of epoch.
        This ensures clean state for next epoch.
        """
        for metric_list in self.metrics.values():
            for met in metric_list:
                if hasattr(met, "metric") and hasattr(met.metric, "reset"):
                    met.metric.reset()

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        confusion_matrix_metric = None
        for met in self.metrics["inference"]:
            if met.name == "confusion_matrix":
                confusion_matrix_metric = met
                break

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                # One media sample per val epoch (first batch; order fixed when shuffle=False)
                if self.writer is not None and batch_idx == 0:
                    self.writer.set_step(epoch * self.epoch_len, part)
                    self._log_batch(batch_idx, batch, part)

            if self.writer is not None:
                self.writer.set_step(epoch * self.epoch_len, part)
                self._log_scalars(self.evaluation_metrics)

        results = self.evaluation_metrics.result()

        for met in self.metrics["inference"]:
            if isinstance(met, EpochMetric):
                extra = met.finalize()
                results.update(extra)
                if self.writer is not None:
                    self.writer.set_step(epoch * self.epoch_len, part)
                    for k, v in extra.items():
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        if fv == fv:  # not NaN
                            self.writer.add_scalar(k, fv)
                met.reset()

        if (
            confusion_matrix_metric is not None
            and hasattr(confusion_matrix_metric, "metric")
            and hasattr(confusion_matrix_metric.metric, "compute")
        ):
            confusion_matrix_data = confusion_matrix_metric.metric.compute()
            if isinstance(confusion_matrix_data, torch.Tensor):
                results["confusion_matrix"] = confusion_matrix_data.detach().cpu()
            else:
                results["confusion_matrix"] = confusion_matrix_data
            confusion_matrix_metric.metric.reset()
            if (
                self.writer is not None
                and getattr(self.writer, "wandb", None) is not None
                and cfg_get(self.config, "writer.log_confusion_matrix_image", True)
                and hasattr(self.writer, "add_confusion_matrix_image")
            ):
                self.writer.add_confusion_matrix_image(
                    "confusion_matrix",
                    results["confusion_matrix"],
                    title=f"{part} epoch {epoch}",
                )

        return results

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
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
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
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
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
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
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
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
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
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
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
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
