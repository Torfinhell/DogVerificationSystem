import json
import random
from pathlib import Path
from typing import List, Tuple, Any, Dict
from torch.utils.data import Dataset
from src.utils.io_utils import ROOT_PATH
class DatasetCombine(Dataset):
    NAME = "DatasetCombine"
    def __init__(
        self,
        datasets: List[Dataset],
        seed: int = 42,
        cache_dir: Path = None,
    ):
        self.datasets = datasets
        self.seed = seed

        if cache_dir is None:
            cache_dir = ROOT_PATH / "data" / "datasets" / "combined_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_names = [getattr(ds, "NAME", f"ds_{i}") for i, ds in enumerate(datasets)]
        self._combined_indices, self.label_mapping = self._load_or_build()
        self.num_classes = len(self.label_mapping)

    def _load_or_build(self) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """Return (combined_indices, label_mapping)."""
        key_data = {
            "dataset_names": self.dataset_names,
            "dataset_lengths": [len(ds) for ds in self.datasets],
            "seed": self.seed,
        }
        cache_file = self.cache_dir / f"combined_{hash(str(key_data))}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            mapping = {tuple(map(int, k.split(','))): v for k, v in data["label_mapping"].items()}
            indices = [tuple(idx) for idx in data["indices"]]
            print(f"Loaded combined data from {cache_file}")
            return indices, mapping
        indices = []
        unique_labels = set()
        label_mapping = {}
        for ds_idx, ds in enumerate(self.datasets):
            ds_index = ds.load_index()
            for item in ds_index:
                if "label" in item:
                    local_label = item["label"]
                    if hasattr(local_label, 'item'):
                        local_label = local_label.item()
                    unique_labels.add((ds_idx, int(local_label)))
        for i, (ds_idx, local_label) in enumerate(sorted(unique_labels)): 
            label_mapping[(ds_idx, local_label)] = i
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                indices.append((ds_idx, sample_idx))

        rng = random.Random(self.seed)
        rng.shuffle(indices)
        cache_data = {
            "indices": indices,
            "label_mapping": {f"{k[0]},{k[1]}": v for k, v in label_mapping.items()},
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved combined data to {cache_file}")

        return indices, label_mapping

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self._combined_indices[idx]
        sample = self.datasets[ds_idx][local_idx]
        if "label" in sample:
            local_label = sample["label"]
            key = (ds_idx, int(local_label))
            if key not in self.label_mapping:
                raise KeyError(f"Label {local_label} from dataset {ds_idx} not found in mapping")
            sample["label"] = self.label_mapping[key]
        return sample

    def __len__(self) -> int:
        return len(self._combined_indices)