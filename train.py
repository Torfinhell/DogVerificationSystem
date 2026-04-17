import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders, get_metrics_and_backends
from src.trainer import Trainer
from src.utils.hydra_cfg import cfg_to_container
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.optim_utils import instantiate_optimizer_and_scheduler
from src.utils.torch_utils import set_tf32_allowance
import os
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """Input: Hydra config. Output: trained model artifacts and logs."""
    set_random_seed(config.trainer.seed)
    if config.get("yandex_token") is not None:
        os.environ["YANDEX_TOKEN"] = config.yandex_token
    project_config = cfg_to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer.logger, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    if config.trainer.get("allow_tf32") is not None:
        set_tf32_allowance(bool(config.trainer.allow_tf32))
    dataloaders, batch_transforms, sampler_criterion = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    
    # Ensure all model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Multi-GPU support - removed DataParallel as it causes gradient issues
    # num_gpus = config.trainer.get("num_gpus", 1)
    # if num_gpus > 1 and device.startswith("cuda"):
    #     logger.info(f"Using {num_gpus} GPUs with DataParallel")
    #     model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    if config.trainer.compile.enabled:
        model = instantiate(config.trainer.compile.call, model)
    training_labels=dataloaders["train"].dataset.get_labels()
    logger.info(model)
    loss_function = instantiate(config.loss_function, labels=training_labels).to(device)
    
    # Ensure all loss function parameters require gradients
    for param in loss_function.parameters():
        param.requires_grad = True
    metrics, backends = get_metrics_and_backends(config, dataloaders, device)
    epoch_len = len(dataloaders["train"])
    optimizer, scheduler = instantiate_optimizer_and_scheduler(config, model, loss_function, epoch_len) #torch.compile is also called here
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        sampler_criterion=sampler_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        backends=backends,
    )

    trainer.train()


if __name__ == "__main__":
    main()
