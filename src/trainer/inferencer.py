import torch
from tqdm.auto import tqdm

from src.metrics.epoch_metric import EpochMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        backend=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
            backend (nn.Module | None): backend for computing similarity scores
                from embeddings (e.g., CosineBackend, PLDABackend).
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.backend = backend

        # For 3D embedding visualization
        self.all_embeddings = []
        self.all_labels = []

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[
                    m.name
                    for m in self.metrics["inference"]
                    if not isinstance(m, EpochMetric)
                ],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        
        # Visualize 3D embeddings
        if self.all_embeddings:
            self._visualize_embeddings_3d()
        
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        # Collect embeddings for visualization
        if "embedding" in batch:
            self.all_embeddings.append(batch["embedding"].detach().cpu())
            if "labels" in batch:
                self.all_labels.append(batch["labels"].detach().cpu())

        # Compute backend scores if backend is available and embeddings exist
        if "embedding" in batch:
            embeddings = batch["embedding"]
            # Always compute cosine scores for comparison
            import torch.nn.functional as F
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            cos_scores = torch.mm(embeddings_norm, embeddings_norm.t())
            batch["cos_scores"] = cos_scores
            
            # Compute backend scores if backend is available
            if self.backend is not None:
                batch_scores = self.backend(embeddings, embeddings)
                batch["backend_scores"] = batch_scores

        if metrics is not None:
            for met in self.metrics["inference"]:
                if isinstance(met, EpochMetric):
                    met(**batch)
                    continue
                met_value = met(**batch)
                if met_value is not None:
                    if isinstance(met_value, torch.Tensor) and met_value.numel() == 1:
                        met_value = float(met_value.detach().cpu().item())
                    metrics.update(met.name, float(met_value))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)

            output_id = current_id + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        results = self.evaluation_metrics.result()
        for met in self.metrics["inference"]:
            if isinstance(met, EpochMetric):
                results.update(met.finalize())
                met.reset()
        return results

    def _visualize_embeddings_3d(self):
        """Visualize collected embeddings in 3D using PCA."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        all_emb = torch.cat(self.all_embeddings, dim=0)
        all_lab = torch.cat(self.all_labels, dim=0) if self.all_labels else None
        
        # Reduce to 3D using PCA
        emb_pca, _, _ = torch.pca_lowrank(all_emb, q=3)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if all_lab is not None:
            colors = all_lab.numpy()
            scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], emb_pca[:, 2], c=colors, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Dog ID')
        else:
            ax.scatter(emb_pca[:, 0], emb_pca[:, 1], emb_pca[:, 2], alpha=0.7)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Embedding Visualization (PCA)')
        
        plt.savefig(self.save_path / "embeddings_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D embedding visualization to {self.save_path / 'embeddings_3d.png'}")
