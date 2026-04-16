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
        random_seed: int = 42,
        cache_dir: Path = None,
    ):
        self.datasets = datasets
        self.seed = random_seed

        if cache_dir is None:
            cache_dir = ROOT_PATH / "data" / "datasets" / "combined_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_names = [getattr(ds, "NAME", f"ds_{i}") for i, ds in enumerate(datasets)]
        self._combined_indices, self.label_mapping = self._load_or_build()
        self.num_classes = len(self.label_mapping)

    def _load_or_build(self) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """Builds shuffled sample pointers and a global label mapping."""
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
        label_mapping = {}
        global_label_counter = 0
        
        for ds_idx, ds in enumerate(self.datasets):
            local_labels = sorted(ds.get_labels())
            for local_label in local_labels:
                key = (ds_idx, int(local_label))
                if key not in label_mapping:
                    label_mapping[key] = global_label_counter
                    global_label_counter += 1

        indices = []
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
        return indices, label_mapping

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single sample with the global mapped label."""
        ds_idx, local_idx = self._combined_indices[idx]
        sample = self.datasets[ds_idx][local_idx]
        
        if "label" in sample:
            local_label = int(sample["label"])
            key = (ds_idx, local_label)
            sample["label"] = self.label_mapping[key]
            
        return sample

    def __len__(self) -> int:
        return len(self._combined_indices)

    def load_index(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all data entries across all datasets, 
        maintaining the shuffled order and applying global label mapping.
        """
        child_indices = [ds.load_index() for ds in self.datasets]
        combined_index = []
        for ds_idx, local_idx in self._combined_indices:
            entry = child_indices[ds_idx][local_idx].copy()
            if "label" in entry:
                local_label = int(entry["label"])
                key = (ds_idx, local_label)
                if key in self.label_mapping:
                    entry["label"] = self.label_mapping[key]
            combined_index.append(entry)
        return combined_index

    def get_labels(self) -> List[int]:
        """
        Returns a list of all global labels, which are guaranteed to be 
        continuous from 0 to (total_unique_labels - 1).
        """
        return sorted(list(self.label_mapping.values()))
