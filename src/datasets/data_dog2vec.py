import json
import random
from pathlib import Path
from typing import List, Dict, Union

import torch
import soundfile as sf
from torch.utils.data import Dataset

from .filter_dog2vec import (
    download_raw_jsons,
    download_info_youtube,
    final_filter_result,
)


class Dog2VecDataset(Dataset):
    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.8,
        data_dir: Union[str, Path] = None,
        random_seed: int = 42,
        *args,
        **kwargs,
    ):
        if part not in ["train", "val"]:
            raise ValueError(f"part must be 'train' or 'val', got {part}")

        self.part = part
        self.train_split = train_split
        self.random_seed = random_seed

        self.data_dir = Path(data_dir or "./data/dog2vec_dataset")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.label_map_path = self.data_dir / "label_map.json"
        self.final_json_path = self.data_dir / "filtered_index.json"

        self._index = self._get_or_create_index()

    def _get_or_create_index(self) -> List[Dict]:
        index_path = self.data_dir / f"{self.part}_index.json"

        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)

        index = self._build_index()

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return index

    def _build_index(self) -> List[Dict]:
        # Ensure pipeline is done
        if not self.final_json_path.exists():
            print("Running Dog2Vec pipeline...")
            download_raw_jsons()
            download_info_youtube()
            final_filter_result()

        with open(self.final_json_path) as f:
            data = json.load(f)

        # Build label map
        breeds = sorted(set(e["label"] for e in data))
        label_map = {b: i for i, b in enumerate(breeds)}

        with open(self.label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)

        # Convert entries
        entries = []
        for e in data:
            path = Path(e["path"])

            if not path.exists():
                continue

            try:
                with sf.SoundFile(path) as f:
                    if f.frames == 0:
                        continue
            except Exception:
                continue

            entries.append(
                {
                    "path": str(path),
                    "label": label_map[e["label"]],
                    "audio_len": float(e["audio_len"]),
                }
            )

        # Shuffle + split
        random.seed(self.random_seed)
        random.shuffle(entries)

        split_idx = int(len(entries) * self.train_split)

        if self.part == "train":
            return entries[:split_idx]
        else:
            return entries[split_idx:]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        entry = self._index[idx]

        audio, sr = sf.read(entry["path"])

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(entry["label"], dtype=torch.long),
            "audio_len": entry["audio_len"],
        }