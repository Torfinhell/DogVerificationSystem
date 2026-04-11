import random
from collections import defaultdict
from torch.utils.data import Dataset

class TestBackend(Dataset):
    def __init__(
        self,
        dataset: Dataset,         
        seed: int,
        part: str = "fit",         
        fit_split: float = 0.9,
    ):
        super().__init__()
        assert part in ["fit", "test"]
        random.seed(seed)
        full_index = dataset.load_index()
        if part == "fit":
            assert all("label" in entry for entry in full_index), \
                "Fit split requires 'label' in each index entry"
        group_by_dog = defaultdict(list)
        for i, entry in enumerate(full_index):
            if "label" in entry:
                group_by_dog[entry["label"]].append(i)  
        if len(group_by_dog):
            self.indices = []
            if part == "fit":
                for indices in group_by_dog.values():
                    split_idx = int(len(indices) * fit_split)
                    random.shuffle(indices)
                    self.indices.extend(indices[:split_idx])
            else:  
                for indices in group_by_dog.values():
                    split_idx = int(len(indices) * fit_split)
                    random.shuffle(indices)
                    self.indices.extend(indices[split_idx:])
        else:
            self.indices=full_index
        random.shuffle(self.indices)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)