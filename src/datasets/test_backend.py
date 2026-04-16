import random
from collections import defaultdict
from torch.utils.data import Dataset

class TestBackend(Dataset):
    def __init__(
        self,
        dataset: Dataset,         
        random_seed: int,
        part: str = "fit",         
        fit_split: float = 0.9,
    ):
        super().__init__()
        assert part in ["fit", "test"]
        random.seed(random_seed)
        full_index = dataset.load_index()
        self.num_classes = getattr(dataset, "num_classes", None)
        
        if part == "fit":
            assert all("label" in entry for entry in full_index), \
                "Fit split requires 'label' in each index entry"
        group_by_dog = defaultdict(list)
        for i, entry in enumerate(full_index):
            if "label" in entry:
                group_by_dog[entry["label"]].append(i)  
        
        if len(group_by_dog):
            self.indices = []
            for _, indices in group_by_dog.items():
                random.shuffle(indices)
                split_idx = int(len(indices) * fit_split)
                
                if part == "fit":
                    self.indices.extend(indices[:split_idx])
                else:  
                    self.indices.extend(indices[split_idx:])
        else:
            self.indices = list(range(len(full_index)))
        
        random.shuffle(self.indices)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def load_index(self):
        """
        Returns the subset of the parent dataset's index 
        corresponding to this split's indices.
        """
        full_index = self.dataset.load_index()
        return [full_index[i] for i in self.indices]

    def get_labels(self):
        """
        Returns a sorted list of unique labels present in this specific split.
        """
        index = self.load_index()
        return sorted(list(set(entry["label"] for entry in index if "label" in entry)))
