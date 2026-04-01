import torch
from torchmetrics.classification import ConfusionMatrix
from src.metrics.epoch_metric import EpochMetric

class ConfusionMatrixMetric(EpochMetric):
    def __init__(self, num_classes, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.device = self._resolve_device(device)
        self.all_preds = []
        self.all_labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        preds = logits.argmax(dim=-1)
        self.all_preds.append(preds.detach().cpu())
        self.all_labels.append(labels.detach().cpu())

    def finalize(self) -> dict:
        if not self.all_preds:
            return {}
        all_preds = torch.cat(self.all_preds)
        all_labels = torch.cat(self.all_labels)
        
        # Get unique labels present in the data
        unique_labels = torch.unique(all_labels).sort().values
        if len(unique_labels) == 0:
            return {}
        
        # Create mapping to 0 to n-1
        label_to_idx = {label.item(): i for i, label in enumerate(unique_labels)}
        num_present_classes = len(unique_labels)
        
        # Remap preds and labels
        remapped_preds = torch.tensor([label_to_idx.get(p.item(), -1) for p in all_preds], dtype=torch.long)
        remapped_labels = torch.tensor([label_to_idx.get(l.item(), -1) for l in all_labels], dtype=torch.long)
        
        # Filter out invalid (shouldn't happen)
        valid = (remapped_preds >= 0) & (remapped_labels >= 0)
        remapped_preds = remapped_preds[valid]
        remapped_labels = remapped_labels[valid]
        
        # Compute confusion matrix
        metric = ConfusionMatrix(num_classes=num_present_classes, task="multiclass").to(self.device)
        matrix = metric(remapped_preds.to(self.device), remapped_labels.to(self.device))
        
        # Return as dict, perhaps save or log elsewhere
        return {"confusion_matrix": matrix.detach().cpu()}

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []