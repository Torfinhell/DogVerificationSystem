from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm


from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.metrics.static_tracker import StaticMetricTracker
from sklearn.metrics import confusion_matrix
from hydra.utils import instantiate
import torch.nn.functional as F
class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        backends,
        optimizer,
        scheduler,
        config,
        device,
        sampler_criterion,
        dataloaders,
        logger,
        writer,
        epoch_len,
        skip_oom=True,
        batch_transforms=None,
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
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer
        self.epoch_len=epoch_len
        self.device = device
        self.skip_oom = skip_oom
        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_transforms = batch_transforms
        self.backends=backends["test_backend"]
        self.sampler_criterion=sampler_criterion
        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        self.lr_scheduler = scheduler
        self.dataloaders=dataloaders
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k in ["val", "test"]
        }
        self.fit_backend_dataloader=dataloaders["fit_backend"]
        self.test_backend_dataloader=dataloaders["test_backend"]
        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

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

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.static_metrics = StaticMetricTracker(
            *self.config.writer.logger.loss_names,
            "grad_norm",
            "loss_grad_norm",
            writer=self.writer
        )
        self.train_metrics = MetricTracker(
            self.metrics["train"],
            writer=self.writer,
        )
        self.val_metrics = MetricTracker(
            self.metrics["val"],
            writer=self.writer,
        )
        self.test_backend_metrics = MetricTracker(
            self.metrics["test_backend"],
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

    def _scheduler_steps_each_batch(self):
        if self.lr_scheduler is None:
            return False
        return self.lr_scheduler.__class__.__name__ in (
            "CyclicLR",
            "OneCycleLR",
            "LinearLR",
            "ExponentialLR",
            "MultiplicativeLR",
            "PolynomialLR",
            "ConstantLR",
        )

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

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"{key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )
            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)
            if stop_process:  # early_stop
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
        self.criterion.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        last_train_metrics={}
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e
            self.static_metrics.update("grad_norm", self._get_grad_norm(self.model))
            self.static_metrics.update("loss_grad_norm", self._get_grad_norm(self.criterion))
            self.static_metrics.update("loss", batch["loss"])
            # log current results
            if batch_idx % self.log_step == 0: #TODO understand what to log when
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_scalars(self.train_metrics)
                self._log_scalars(self.static_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                last_static_metrics=self.static_metrics.result()
                self.train_metrics.reset()
                self.static_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break
        if self.lr_scheduler is not None and not self._scheduler_steps_each_batch():
            self.lr_scheduler.step()
        logs={f"train_{name}": value for name, value in last_train_metrics.items()}
        logs.update({name: value for name, value in last_static_metrics.items()})
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})
        if self.backends is not None:
            embeddings_tensor, labels_tensor=self.collect_embeddings_lables(self.fit_backend_dataloader)
            for backend in self.backends:
                self.fit_backend(embeddings_tensor, labels_tensor, backend)
                backend_logs=self.evaluate_backend(epoch, backend, dataloader=self.test_backend_dataloader)
                logs.update(**{f"test_{backend.NAME}_{name}": value for name, value in backend_logs.items()})
        if self.sampler_criterion is not None:
            normalized_weight = F.normalize(self.criterion.weight.data, dim=1)
            sm = torch.matmul(normalized_weight, normalized_weight.T)
            self.writer.add_similarity_matrix_image("similiarity_matrix", sm, title="Similiarity matrix from AAM prototypes")
            self.sampler_criterion.sm = sm
        self.log_after_epoch(epoch)
        return logs

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
        self.criterion.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                try:
                    batch = self.process_batch(
                        batch,
                        metrics=self.val_metrics,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        torch.cuda.empty_cache()  # free some memory
                        continue
                    else:
                        raise e
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.val_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.val_metrics.result()
    def collect_embeddings_lables(self, dataloader, labels_key="label"):
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc="collecting embeddings...:", total=len(dataloader)):
                try:
                    batch = self.process_batch(
                        batch,
                        metrics=None,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        torch.cuda.empty_cache()  # free some memory
                        continue
                    else:
                        raise e
                all_embeddings.append(batch["embedding"].detach().cpu())
                if labels_key in batch:
                    all_labels.append(batch[labels_key].detach().cpu())
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None
        return embeddings_tensor, labels_tensor
    def fit_backend(self, embeddings_tensor, labels_tensor, backend):
        self.is_train = False
        self.model.eval()
        self.criterion.eval()
        backend.reset()
        backend.fit(embeddings_tensor, labels_tensor)
    def evaluate_backend(self,epoch,backend, dataloader):
        self.is_train = False
        self.model.eval()
        self.criterion.eval()
        self.test_backend_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=f"running test: {backend.NAME}",
                total=len(dataloader),
            ):
                try:
                    batch = self.process_batch(
                        batch,
                        metrics=self.test_backend_metrics,
                        backend=backend
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        torch.cuda.empty_cache()  # free some memory
                        continue
                    else:
                        raise e
            self.writer.set_step(epoch * self.epoch_len, f"test_{backend.NAME}") #TODo fix self.epoch_len=None
            self._log_scalars(self.test_backend_metrics)
            self._log_batch(
                batch_idx, batch
            )  # log only the last batch during inference
        return self.test_backend_metrics.result()
    def log_after_epoch(self, epoch):
        self.writer.set_step(epoch * self.epoch_len, "")
        for ds_name, key in self.config.writer.plot_3d_ds:
            dataloader=self.dataloaders[ds_name]
            embeddings, labels=self.collect_embeddings_lables(dataloader, labels_key=key)
            self.writer.add_plot_3d(
                plot_name=f"{ds_name}_plot_3d",
                embeddings=embeddings,
                labels=labels,
                title=f"Visualizing embeddings for {ds_name}"
            )
        for backend in self.backends:
            for key in self.config.writer.confusion_matrix.key_labels:
                dataloader=self.dataloaders["test_backend"]
                index_part=dataloader.dataset.load_index()
                labels_to_key=dict(((int(entry["label"]), entry[key]) for entry in index_part))
                gt_keys=[]
                pred_keys=[]
                with torch.no_grad():
                    for batch_idx, batch in tqdm(
                        enumerate(dataloader),
                        desc=f"collecting data from confusion_matrix by key: {key}",
                        total=len(dataloader),
                    ):
                        try:
                            batch = self.process_batch(
                                batch,
                                metrics=self.test_backend_metrics,
                                backend=backend
                            )
                        except torch.cuda.OutOfMemoryError as e:
                            if self.skip_oom:
                                self.logger.warning("OOM on batch. Skipping batch.")
                                torch.cuda.empty_cache()  # free some memory
                                continue
                            else:
                                raise e
                        gt_keys.extend([int(x) for x in batch[key].cpu().numpy()])
                        pred_keys.extend([labels_to_key[int(label_pred)] for label_pred in batch["pred"]])
                all_keys=list(set(gt_keys)|set(pred_keys))
                self.writer.add_confusion_matrix_image(
                    name=f"{backend.NAME}_{key}_confusion_matrix",
                    y_true=gt_keys,
                    y_pred=pred_keys,
                    labels=all_keys)
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
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
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
        # do batch transforms on device
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
    def _get_grad_norm(self, module, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = module.parameters()
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
        for metric_name, res_metric in metric_tracker.result().items():
            self.writer.add_scalar(f"{metric_name}", res_metric)

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

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
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
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)