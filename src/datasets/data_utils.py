from copy import deepcopy
from itertools import repeat
from hydra.utils import instantiate
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
from collections import defaultdict
from copy import deepcopy
from omegaconf import ListConfig, DictConfig, OmegaConf
def _set_key_on_metric_cfg(metric_cfg, key_name, key_value):
    def _recursive_update(item):
        if isinstance(item, (list, ListConfig)):
            for i in item:
                _recursive_update(i)
        elif isinstance(item, (dict, DictConfig)):
            if isinstance(item, DictConfig):
                OmegaConf.set_readonly(item, False)
            for key, value in item.items():
                if key == key_name:
                    item[key] = key_value
                else:
                    _recursive_update(value)
    _recursive_update(metric_cfg)
    return metric_cfg
def get_metrics_and_backends(config, dataloaders, device):
    """Input: config and dataloaders. Output: instantiated metrics and backends."""
    metrics = {} 
    for part_name, cfg_metric in config.metrics.items():
        if part_name == "classification_metrics": 
            continue
        num_classes = dataloaders[part_name].dataset.num_classes
        labels=dataloaders[part_name].dataset.get_labels()
        _set_key_on_metric_cfg(cfg_metric, "num_classes", num_classes)
        _set_key_on_metric_cfg(cfg_metric, "labels", labels)
        metrics[part_name] = instantiate(cfg_metric)
    backends = {}
    if config.get("backends") is not None:
        for part_name, part_backends_cfg in config.backends.items():
            if part_backends_cfg is None: 
                backends[part_name]=[]
                continue
            labels=dataloaders[part_name].dataset.get_labels()
            _set_key_on_metric_cfg(part_backends_cfg, "labels",labels)
            backends[part_name]=[instantiate(cfg_backend)  for cfg_backend in part_backends_cfg]      
    return metrics, backends

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
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)
    datasets = instantiate(config.datasets)
    dataloaders = {}
    sampler_criterion = None

    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]
        is_train = (dataset_partition == "train")
        loader_kwargs = {
            "dataset": dataset,
            "collate_fn": collate_fn,
            "worker_init_fn": set_worker_seed,
        }
        if config.get("batch_sampler", None) is not None and is_train:
            batch_sampler = instantiate(
                config.batch_sampler.sampler,
                ds=dataset,
            )
            sampler_criterion = batch_sampler.criterion
            dataloaders[dataset_partition] = instantiate(
                config.dataloader.dataloader_with_batch_sampler, 
                batch_sampler=batch_sampler,
                **loader_kwargs
            )
        else:
            dataloaders[dataset_partition] = instantiate(
                config.dataloader.dataloader_standard,
                shuffle=is_train,
                drop_last=is_train,
                **loader_kwargs
            )

    return dataloaders, batch_transforms, sampler_criterion
