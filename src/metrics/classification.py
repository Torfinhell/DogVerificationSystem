import inspect
import torch

from src.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    def __init__(self, classification_metric, device, *args, **kwargs):
        """Input: torchmetrics object and device. Output: wrapped metric."""
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._metric_cls = classification_metric.__class__
        self._metric_kwargs = self._extract_metric_init_kwargs(classification_metric)
        self._num_classes = getattr(classification_metric, "num_classes", None)
        self.classification_metric = self._recreate_metric()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        """Input: class count. Output: metric recreated with new class count."""
        self._num_classes = value
        if hasattr(self.classification_metric, "num_classes"):
            self._metric_kwargs["num_classes"] = value
            self.classification_metric = self._recreate_metric()

    def _extract_metric_init_kwargs(self, metric):
        signature = inspect.signature(metric.__class__.__init__)
        kwargs = {}
        for name, param in signature.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if hasattr(metric, name):
                kwargs[name] = getattr(metric, name)
            elif param.default is not inspect._empty:
                kwargs[name] = param.default
        return kwargs

    def _recreate_metric(self):
        if self._num_classes is not None:
            self._metric_kwargs["num_classes"] = self._num_classes
        metric = self._metric_cls(**self._metric_kwargs)
        return metric.to(self.device)

    def __call__(
        self,
        logits: torch.Tensor | None = None,
        preds: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        **kwargs,
    ):
        """Input: logits/preds and labels. Output: scalar metric value."""
        if logits is None and preds is None:
            logits = kwargs.get("logits")
            preds = kwargs.get("preds")
        if label is None:
            label = kwargs.get("label")

        assert label is not None, "ClassificationMetric requires a target label tensor"

        label = label.to(self.device)

        if preds is not None:
            preds = preds.to(self.device)
            if preds.dim() == 1:
                classes = preds
            else:
                assert preds.dim() == 2, "ClassificationMetric expects preds with 1 or 2 dims"
                assert (
                    self.num_classes is None or preds.shape[-1] == self.num_classes
                ), (
                    f"ClassificationMetric received preds with shape {preds.shape} but expected "
                    f"num_classes={self.num_classes} for class scores. If preds are embeddings, "
                    f"pass explicit label predictions instead of embeddings."
                )
                classes = preds.argmax(dim=-1)
        else:
            assert logits is not None, "ClassificationMetric requires either logits or preds"
            logits = logits.to(self.device)
            if logits.dim() == 1:
                classes = logits
            else:
                assert logits.dim() == 2, "ClassificationMetric expects logits with 1 or 2 dims"
                assert (
                    self.num_classes is None or logits.shape[-1] == self.num_classes
                ), (
                    f"ClassificationMetric received logits with shape {logits.shape} that do not match "
                    f"num_classes={self.num_classes}. This likely means the model is returning embeddings "
                    f"as logits. Use explicit class predictions or remove the classification metric."
                )
                classes = logits.argmax(dim=-1)

        assert classes.shape == label.shape, (
            f"ClassificationMetric got mismatched shapes: classes {classes.shape} vs label {label.shape}"
        )

        if self.num_classes is not None and classes.numel() > 0 and label.numel() > 0:
            required_num_classes = max(classes.max().item(), label.max().item()) + 1
            if required_num_classes > self.num_classes:
                if getattr(self.classification_metric, "_update_count", 0) == 0:
                    self.num_classes = required_num_classes
                else:
                    raise AssertionError(
                        f"ClassificationMetric got label indices in range [0, {label.max().item()}] "
                        f"or predicted class indices in range [0, {classes.max().item()}] "
                        f"but num_classes={self.num_classes} was configured."
                    )

        assert (
            self.num_classes is None
            or classes.numel() == 0
            or label.numel() == 0
            or max(classes.max(), label.max()) < self.num_classes
        ), (
            f"ClassificationMetric got class/label indices in range [0, {max(classes.max().item(), label.max().item())}] "
            f"but num_classes={self.num_classes} was configured."
        )

        self.classification_metric.update(classes, label)
        result = self.classification_metric.compute()
        return float(result.detach().cpu().item())

