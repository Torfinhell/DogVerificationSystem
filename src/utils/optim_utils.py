from __future__ import annotations

import torch.nn as nn
from hydra.utils import instantiate


def instantiate_optimizer(
    config_optimizer,
    model: nn.Module,
    loss_module: nn.Module,
):
    """
    Build the optimizer with optional separate learning rate for loss parameters
    (e.g. AAM-Softmax ``nn.Embedding``).

    Config keys:
        ``lr``: learning rate for model parameters.
        ``lr_loss``: if set (not ``None``) and the loss has trainable parameters,
        those parameters use this lr; otherwise all trainable params use ``lr``.
    """
    target = config_optimizer._target_
    lr = config_optimizer.lr
    lr_loss = config_optimizer.get("lr_loss")

    exclude = {"_target_", "lr", "lr_loss"}
    extra = {k: config_optimizer[k] for k in config_optimizer if k not in exclude}

    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in loss_module.parameters() if p.requires_grad]

    if loss_params and lr_loss is not None:
        param_groups = [
            {"params": model_params, "lr": lr},
            {"params": loss_params, "lr": lr_loss},
        ]
        return instantiate({"_target_": target, **extra}, params=param_groups)

    return instantiate(
        {"_target_": target, "lr": lr, **extra},
        params=model_params + loss_params,
    )
