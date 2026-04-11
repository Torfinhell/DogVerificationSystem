import json
import shutil
import random
from pathlib import Path
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os
from src.utils.io_utils import ROOT_PATH
from .base_dataset import BaseDataset
class BarkopediaDataset(BaseDataset):
    NAME="Barkopedia"
    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.7,
        data_dir: Path = None,
        random_seed: int = 42,
        hub_workers: int = 1,
        *args,
        **kwargs,
    ):
        assert part in ["train", "val", "test"]
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "barkopedia"
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.part = part
        self.random_seed = random_seed
        self.hub_workers = hub_workers
        index = self.load_index()
        super().__init__(index, *args, **kwargs)

    def load_index(self):
        """Load index from JSON if exists, otherwise create it."""
        index_path = self.data_dir / f"{self.part}_index.json"
        if not index_path.exists():
            self.build_indices()
        with index_path.open() as f:
            index = json.load(f)
        return index

    def build_indices(self):
        split_to_download = "train" if self.part != "test" else "test"
        part_data_dir = self.data_dir / split_to_download
        audio_dir = part_data_dir / "audio"
        metadata_path = part_data_dir / "metadata.csv"
        if not audio_dir.exists() or not metadata_path.exists():
            self._download_split(split_to_download, part_data_dir)
        df = pd.read_csv(metadata_path)
        if self.part != "test":
            unique_dog_ids = list(sorted(df['dog_id'].unique()))
            mapping_path = self.data_dir / "mapping.json"
            if mapping_path.exists():
                with mapping_path.open() as f:
                    mapping = json.load(f)
                dog_id_to_class = {int(k): int(v) for k, v in mapping["dog_id_to_class"].items()}
            else:
                dog_id_to_class = {int(dog_id): i for i, dog_id in enumerate(unique_dog_ids)}
                mapping = {
                    "num_classes": len(dog_id_to_class),
                    "dog_id_to_class": {str(k): v for k, v in dog_id_to_class.items()},
                }
                with mapping_path.open("w") as f:
                    json.dump(mapping, f, indent=2)
            random.seed(self.random_seed)
            random.shuffle(unique_dog_ids)
            train_idx = int(len(unique_dog_ids) * self.train_split)
            if self.part == "train":
                dog_ids = set(unique_dog_ids[:train_idx])
            else:  
                dog_ids = set(unique_dog_ids[train_idx:])
        else:
            dog_ids = None 
            unique_dog_ids = [] 
            train_idx = 0
        entries = []
        for _, row in df.iterrows():
            dog_id = row.get("dog_id", None)
            if self.part != "test" and dog_id is not None and dog_id not in dog_ids:
                continue
            audio_path = audio_dir / row["filename"]
            if not audio_path.exists():
                print(f"Warning: file {audio_path} missing, skipping")
                continue
            if audio_path.stat().st_size == 0:
                print(f"Warning: empty file {audio_path}, skipping")
                continue
            with sf.SoundFile(audio_path) as f:
                if f.frames == 0:
                    print(f"Warning: file {audio_path} has zero frames, skipping")
                    continue

            if self.part != "test":
                entry = {
                    "path": str(audio_path.absolute()),
                    "audio_len": float(row["duration"]),
                    "label": dog_id_to_class[dog_id],
                }
            else:
                entry = {
                    "path": str(audio_path.absolute()),
                    "audio_len": float(row["duration"]),
                }
            entries.append(entry)

        index_path = self.data_dir / f"{self.part}_index.json"
        with open(index_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Saved {self.part} index ({len(entries)} entries) to {index_path}")

    def _download_split(self, split: str, target_dir: Path):
        print(f"Downloading split '{split}' from Hugging Face via snapshot_download...")
        token = os.environ.get("HF_TOKEN", None)
        temp_dir = target_dir / "temp_download"
        snapshot_download(
            repo_id="ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset",
            repo_type="dataset",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=self.hub_workers, 
            ignore_patterns=["*.parquet", "*.json", "*.csv", "*.txt", "*.md", "*.yaml"],
        )
        audio_dir = target_dir / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        metadata = []

        split_temp = temp_dir / split
        search_dir = split_temp if split_temp.exists() else temp_dir

        for audio_path in tqdm(
            sorted(search_dir.glob("**/*.wav")) +
            sorted(search_dir.glob("**/*.mp3")) +
            sorted(search_dir.glob("**/*.flac")),
            desc=f"Processing {split} audio"
        ):
            try:
                audio_id = audio_path.stem
                original_ext = audio_path.suffix
                filename = f"{audio_id}{original_ext}"

                dst_path = audio_dir / filename
                shutil.copy2(audio_path, dst_path)

                info = sf.info(str(audio_path))
                duration = info.duration
                row = {
                    "audio_id": audio_id,
                    "filename": filename,
                    "duration": duration,
                }
                if split == "train":
                    row.update({"dog_id": int(audio_path.parent.name)})
                metadata.append(row)
            except Exception as e:
                print(f"Warning: error processing {audio_path}: {e}, skipping")
                continue

        shutil.rmtree(temp_dir, ignore_errors=True)
        df = pd.DataFrame(metadata)
        csv_path = target_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(metadata)} entries to {csv_path}")