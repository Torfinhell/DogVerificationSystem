import torch
from torchmetrics.classification import ConfusionMatrix
from src.metrics.epoch_metric import EpochMetric

class ConfusionMatrixMetric(EpochMetric):
    def __init__(self, num_classes, device, metric_type="dog_id", *args, **kwargs):
        """
        Compute confusion matrix for classification tasks.
        
        Args:
            num_classes: Number of classes
            device: Device
            metric_type: "dog_id" (predictions vs labels) or "breed" (breed predictions vs labels)
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.device = self._resolve_device(device)
        self.metric_type = metric_type
        self.all_predictions = []
        self.all_labels = []
        self.all_breeds = []

    def __call__(self, preds: torch.Tensor | None = None, batch_dict: dict | None = None, **batch):
        """
        Compute confusion matrix for classification predictions.
        
        Args:
            preds: Predicted logits or class labels [batch_size] or [batch_size, num_classes]
            batch_dict: Dictionary containing batch data with keys:
                       - "dog_id": Ground truth dog identity labels
                       - "breed": Breed labels
            **batch: Additional batch data fields passed by the trainer.
        """
        if batch_dict is None:
            batch_dict = batch
        if preds is None:
            preds = batch.get("preds")

        if preds is None or batch_dict is None:
            return

        # Handle logits (argmax to get class predictions)
        if preds.dim() > 1:
            preds = torch.argmax(preds, dim=1)

        # Extract labels based on metric_type
        labels = batch_dict.get(self.metric_type)

        if labels is None:
            return

        if self.metric_type == "dog_id":
            self.all_predictions.append(preds.detach().cpu())
            self.all_labels.append(labels.detach().cpu())
        elif self.metric_type == "breed":
            self.all_breeds.append(preds.detach().cpu())
            self.all_labels.append(labels.detach().cpu())

    def finalize(self) -> dict:
        if self.metric_type == "dog_id":
            return self._finalize_dog_id()
        elif self.metric_type == "breed":
            return self._finalize_breed()
        return {}

    def _finalize_dog_id(self) -> dict:
        """Compute confusion matrix for dog identity classification."""
        if not self.all_predictions or not self.all_labels:
            return {}
        
        all_predictions = torch.cat(self.all_predictions)  # [N]
        all_labels = torch.cat(self.all_labels)  # [N]
        
        # Compute confusion matrix
        metric = ConfusionMatrix(num_classes=self.num_classes, task="multiclass").to(self.device)
        matrix = metric(all_predictions.to(self.device), all_labels.to(self.device))
        
        return {f"{self.name}": matrix.detach().cpu()}

    def _finalize_breed(self) -> dict:
        """Compute confusion matrix for breed classification."""
        if not self.all_breeds or not self.all_labels:
            return {}
        
        all_breeds = torch.cat(self.all_breeds)
        all_labels = torch.cat(self.all_labels)
        
        # Get unique breed classes in the dataset
        unique_breeds = torch.unique(all_breeds).sort().values
        num_breed_classes = len(unique_breeds)
        
        if num_breed_classes == 0:
            return {}
        
        # Compute confusion matrix
        metric = ConfusionMatrix(num_classes=num_breed_classes, task="multiclass").to(self.device)
        matrix = metric(all_breeds.to(self.device), all_labels.to(self.device))
        
        return {f"{self.name}": matrix.detach().cpu()}

    def reset(self) -> None:
        self.all_predictions = []
        self.all_labels = []
        self.all_breeds = []