import logging
import random
from typing import List, Optional, Callable, Dict
import torch
from torch.utils.data import Dataset
import soundfile as sf

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index: List[dict],
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms: Optional[Dict[str, Callable]] = None,
    ):
        self._assert_index_is_valid(index)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index = index
        self.instance_transforms = instance_transforms
        self.num_classes = None
        if any("label" in e for e in self._index):
            self.num_classes = len({e["label"] for e in self._index if "label" in e})

    def load_index(self) -> List[dict]:
        """Override in subclasses to build/load index from disk."""
        raise NotImplementedError

    def __getitem__(self, ind: int):
        data_dict = self._index[ind]
        audio, sr = sf.read(data_dict["path"])
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)          
        else:
            audio = audio.transpose(0, 1) 
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        instance_data = {
            "audio": audio,
            "sample_rate": sr,
        }
        if "label" in data_dict:
            instance_data["label"] = data_dict["label"]
        if "breed" in data_dict:
            instance_data["breed"] = data_dict["breed"]
        instance_data = self.preprocess_data(instance_data)
        if "audio" in instance_data:
            instance_data["audio"] = instance_data["audio"].squeeze()
        if self.instance_transforms and "get_spectral_feat" in self.instance_transforms:
            spectral_feat = self.instance_transforms["get_spectral_feat"](instance_data["audio"])
            instance_data.update(self.preprocess_data({"spectral_feat": spectral_feat}))
        return instance_data
    def __len__(self):
        return len(self._index)
    def preprocess_data(self, instance_data):
        if self.instance_transforms:
            for name, func in self.instance_transforms.items():
                if func and name != "get_spectral_feat" and name in instance_data:
                    instance_data[name] = func(instance_data[name])
        return instance_data
    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "Each entry must have 'path' key"
    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index