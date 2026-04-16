import torch.nn as nn
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

def instantiate_optimizer_and_scheduler(
    config,
    model: nn.Module,
    loss_module: nn.Module,
    epoch_len,
):
    if config.trainer.compile.enabled:
        model = instantiate(config.trainer.compile.call, model)
        loss_module = instantiate(config.trainer.compile.call, loss_module)

    lr = float(config.max_lr)
    lr_loss = float(config.get("max_lr_loss")) if config.get("max_lr_loss") is not None else None
    base_lr = float(config.base_lr)
    base_lr_loss = float(config.base_lr_loss)

    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in loss_module.parameters() if p.requires_grad]

    optimizer_config = config.get("optimizer")
    if optimizer_config is None:
        raise ValueError("Optimizer config is missing.")

    opt_config = OmegaConf.to_container(optimizer_config, resolve=True)
    opt_target = opt_config.pop("_target_")
    opt_class = get_class(opt_target)
    opt_config.pop("params", None)
    opt_config.pop("lr", None)

    if loss_params and lr_loss is not None:
        param_groups = [
            {"params": model_params, "lr": lr},
            {"params": loss_params, "lr": lr_loss},
        ]
        optimizer = opt_class(param_groups, **opt_config)

        base_lr_loss = (lr_loss / lr) * base_lr
        sched_base = [base_lr, base_lr_loss]
        sched_max = [lr, lr_loss]
    else:
        params = model_params + loss_params
        optimizer = opt_class(params, lr=lr, **opt_config)
        sched_base = base_lr
        sched_max = lr

    scheduler = None
    scheduler_config = config.get("scheduler") or config.get("lr_scheduler")
    if scheduler_config is not None:
        sched_cfg = OmegaConf.to_container(scheduler_config, resolve=True)
        sched_target = sched_cfg.pop("_target_")
        sched_class = get_class(sched_target)

        unsupported_keys = {
            "optimizer",
            "base_lr",
            "max_lr",
            "step_size_up",
            "epochs",
            "steps_per_epoch",
            "total_steps",
        }
        scheduler_name = sched_class.__name__
        if scheduler_name != "CosineAnnealingLR":
            unsupported_keys.add("T_max")
        sched_args = {k: v for k, v in sched_cfg.items() if k not in unsupported_keys}

        if scheduler_name == "CyclicLR":
            scheduler = sched_class(
                optimizer,
                base_lr=sched_base,
                max_lr=sched_max,
                step_size_up=epoch_len,
                **sched_args,
            )
        elif scheduler_name == "OneCycleLR":
            if "max_lr" not in sched_cfg:
                sched_args["max_lr"] = sched_max
            if (
                "total_steps" not in sched_cfg
                and "epochs" not in sched_cfg
                and "steps_per_epoch" not in sched_cfg
            ):
                sched_args["epochs"] = config.trainer.n_epochs
                sched_args["steps_per_epoch"] = epoch_len
            scheduler = sched_class(optimizer, **sched_args)
        elif scheduler_name == "CosineAnnealingLR":
            if "T_max" not in sched_cfg:
                sched_args["T_max"] = config.trainer.n_epochs
            scheduler = sched_class(optimizer, **sched_args)
        else:
            scheduler = sched_class(optimizer, **sched_args)

    return optimizer, scheduler
