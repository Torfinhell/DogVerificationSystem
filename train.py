import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.hydra_cfg import cfg_to_container
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.optim_utils import instantiate_optimizer
from src.utils.torch_utils import set_tf32_allowance
import os 

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    if config.get("yandex_token") is not None:
        os.environ["YANDEX_TOKEN"] = config.yandex_token
    #add here extraction of yandex_token by config.yandex_token and making it env variable
    project_config = cfg_to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer.logger, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    if config.trainer.get("allow_tf32") is not None:
        set_tf32_allowance(bool(config.trainer.allow_tf32))

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture; optional torch.compile via Hydra (trainer.compile.call)
    model = instantiate(config.model).to(device)
    if config.trainer.compile.enabled:
        model = instantiate(config.trainer.compile.call, model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # instantiate backend for verification (used during validation/inference)
    backend = None
    if config.get("backends") is not None:
        backend = instantiate(config.backends).to(device)
        logger.info(f"Loaded backend: {backend.__class__.__name__}")

    # optimizer: optional lr_loss for AAM-Softmax (etc.) vs model — see optim_utils
    if(config.trainer.epoch_len is None):
        config.trainer.epoch_len=len(dataloaders["train"].dataset)
    optimizer = instantiate_optimizer(config.optimizer, model, loss_function)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        backend=backend,
    )

    trainer.train()


if __name__ == "__main__":
    main()
