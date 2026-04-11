from copy import deepcopy
from itertools import repeat
from hydra.utils import instantiate
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

def _non_epoch_metric_names(metric_list):
    from src.metrics.epoch_metric import EpochMetric

    return [m.name for m in metric_list if not isinstance(m, EpochMetric)]


def _build_backend_keys(backends):
    backend_name_counts = {}
    backend_keys = []
    for backend in backends:
        base_name = backend.__class__.__name__
        count = backend_name_counts.get(base_name, 0) + 1
        backend_name_counts[base_name] = count
        backend_keys.append(base_name if count == 1 else f"{base_name}_{count}")
    return backend_keys


def metric_keys_for_partition(metrics, part_name):
    """Input: metrics dict, partition name. Output: non-epoch metric names."""
    from src.metrics.epoch_metric import EpochMetric

    partition_metrics = metrics.get(part_name, [])
    if isinstance(partition_metrics, dict):
        partition_metrics = partition_metrics.get("main", [])
    return [m.name for m in partition_metrics if not isinstance(m, EpochMetric)]


def _set_num_classes_on_metric_cfg(metric_cfg, num_classes):
    if not hasattr(metric_cfg, "get"):
        return
    if metric_cfg.get("_target_") == "src.metrics.ClassificationMetric":
        classification_metric_cfg = metric_cfg.get("classification_metric")
        if classification_metric_cfg is not None and hasattr(classification_metric_cfg, "__setitem__"):
            classification_metric_cfg["num_classes"] = num_classes
    if "num_classes" in metric_cfg:
        metric_cfg["num_classes"] = num_classes


def get_metrics_and_backends(config, dataloaders):
    """Input: config and dataloaders. Output: instantiated metrics and backends."""
    metrics_cfg = deepcopy(config.metrics)

    for part in ["val", "test"]:
        if part in dataloaders and part in metrics_cfg:
            num_classes = dataloaders[part].dataset.num_classes
            for met_cfg in metrics_cfg[part]:
                _set_num_classes_on_metric_cfg(met_cfg, num_classes)

    metrics = instantiate(metrics_cfg)
    backends = []
    if config.get("backends") is not None:
        backends = instantiate(config.backends)
    backend_keys = _build_backend_keys(backends)
    train_metric_objects = metrics.get("train", [])
    val_metric_objects = metrics.get("val", [])
    test_metric_objects = metrics.get("test", [])
    backend_metric_objects = {}
    backend_metric_names = {}
    if backends and isinstance(test_metric_objects, list):
        for backend_key in backend_keys:
            per_backend_metrics = deepcopy(test_metric_objects)
            backend_metric_objects[backend_key] = per_backend_metrics
            backend_metric_names[backend_key] = _non_epoch_metric_names(per_backend_metrics)
    prepared_metrics = {
        "train": train_metric_objects,
        "val": val_metric_objects,
        "test": {"backends": backend_metric_objects},
        "metric_keys": {
            "train": _non_epoch_metric_names(train_metric_objects),
            "val": _non_epoch_metric_names(val_metric_objects),
            "test_backends": backend_metric_names,
        },
        "backend_keys": backend_keys,
    }
    if "inference" in metrics:
        prepared_metrics["inference"] = metrics["inference"]
        prepared_metrics["metric_keys"]["inference"] = _non_epoch_metric_names(metrics["inference"])
    return prepared_metrics, backends

def inf_loop(dataloader):
    """Input: finite dataloader. Output: endless dataloader iterator."""
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """Input: transform mapping and device. Output: transforms moved to device."""
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """Input: config and device. Output: dataloaders and batch transforms."""
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)
    datasets = instantiate(config.datasets)
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.dataloader_standard.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.dataloader_standard.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        dataloader_kwargs = {
            "dataset": dataset,
            "collate_fn": collate_fn,
            "drop_last": (dataset_partition == "train"),
            "worker_init_fn": set_worker_seed,
        }
        dataloader_kwargs["shuffle"] = (dataset_partition == "train")
        partition_dataloader = instantiate(config.dataloader.dataloader_standard, **dataloader_kwargs)
        dataloaders[dataset_partition] = partition_dataloader
    return dataloaders, batch_transforms


