import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def fit_backend_on_dataset(model, backend, config, device, dataset_name):
    """Input: model, backend, config, dataset name. Output: fitted backend."""
    if backend is None or not hasattr(backend, "fit"):
        return
    print(f"\nCollecting embeddings from '{dataset_name}' dataset for backend training...")
    temp_config = OmegaConf.create(config)
    temp_config.datasets = dataset_name
    try:
        dataloaders_train, batch_transforms_train = get_dataloaders(temp_config, device)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}' for backend training: {e}")
        print("Skipping backend fitting.")
        return
    embeddings_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for part, dataloader in dataloaders_train.items():
            for batch in tqdm(
                dataloader,
                desc=f"Collecting embeddings from {part}",
                total=len(dataloader),
            ):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                if batch_transforms_train and part in batch_transforms_train:
                    for transform in batch_transforms_train[part]:
                        batch = transform(batch)
                with torch.no_grad():
                    outputs = model(**batch)
                emb = outputs.get("embedding")
                if emb is None:
                    emb = outputs.get("logits")
                if emb is not None:
                    embeddings_list.append(emb.detach().cpu())
                    if "label" in batch:
                        labels_list.append(batch["label"].detach().cpu())
    all_embeddings = torch.cat(embeddings_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0) if labels_list else None
    backend.fit(all_embeddings.to(device), labels=all_labels)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """Input: Hydra config. Output: inference metrics and saved predictions."""
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    print(model)

    metrics = instantiate(config.metrics)
    backend = None
    if config.get("backends") is not None:
        backend = instantiate(config.backends).to(device)
        print(f"Loaded backend: {backend.__class__.__name__}")
        train_backend_dataset = config.inferencer.get("train_backend_on_dataset")
        if train_backend_dataset is not None:
            fit_backend_on_dataset(model, backend, config, device, train_backend_dataset)
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
        backend=backend,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
