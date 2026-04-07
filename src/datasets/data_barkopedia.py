import json
import shutil
import random
from pathlib import Path
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class BarkopediaDataset(BaseDataset):
    """
    Barkopedia Individual Dog Recognition Dataset.

    Downloads from Hugging Face, saves audio files preserving their original format,
    and builds indexes (JSON) for train, validation, and test splits.
    The train and validation splits are created by splitting **each dog_id's samples**
    according to train_split (instead of splitting whole dog IDs). If train_split is None,
    all samples are assigned to the training split and the validation split is empty.
    """

    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        data_dir: Path = None,
        num_classes: int | None = None,
        random_seed: int = 42,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        if part not in ["train", "val", "test", "val_full"]:
            raise ValueError(f"Part must be 'train', 'val', 'test', or 'val_full', got '{part}'")

        # Validate splits sum to 1.0
        total_split = train_split + val_split + test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small float tolerance
            raise ValueError(f"train_split + val_split + test_split must equal 1.0, got {total_split}")

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "barkopedia"
        self._data_dir = Path(data_dir)
        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split
        self._part = part
        self._random_seed = random_seed
        self.sample_rate = sample_rate

        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        """Load index from JSON if exists, otherwise create it."""
        index_path = self._data_dir / f"{self._part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            if self._part in ("train", "val", "val_full"):
                self._build_train_val_indices()
            else:
                self._build_test_index()
            with index_path.open() as f:
                index = json.load(f)
        return index

    def _build_train_val_indices(self):
        """
        Build and save train, val, test indices based on dog ID stratification.
        Dog IDs are split into train/val/test sets according to respective split proportions.
        All samples from each dog ID go to the corresponding split.
        val_full combines val and test for backend training.
        """
        train_dir = self._data_dir / "train"
        audio_dir = train_dir / "audio"
        metadata_path = train_dir / "metadata.csv"

        if not audio_dir.exists() or not metadata_path.exists():
            self._download_split("train", train_dir)
        df = pd.read_csv(metadata_path)
        mapping_path = self._data_dir / "mapping.json"
        if mapping_path.exists():
            with mapping_path.open() as f:
                mapping = json.load(f)
            dog_id_to_class = {int(k): int(v) for k, v in mapping["dog_id_to_class"].items()}
        else:
            unique_dog_ids = sorted(df['dog_id'].unique())
            dog_id_to_class = {int(did):i for i, did in enumerate(unique_dog_ids)}
            mapping = {
                "num_classes": len(dog_id_to_class),
                "dog_id_to_class": {str(k): v for k, v in dog_id_to_class.items()},
            }
            with mapping_path.open("w") as f:
                json.dump(mapping, f, indent=2)
        
        unique_dog_ids = sorted(df['dog_id'].unique())
        random_gen = random.Random(self._random_seed)
        shuffled_dog_ids = list(unique_dog_ids)
        random_gen.shuffle(shuffled_dog_ids)
        
        # Calculate split indices
        train_idx = int(len(shuffled_dog_ids) * self._train_split)
        val_idx = train_idx + int(len(shuffled_dog_ids) * self._val_split)
        
        train_dog_ids = set(shuffled_dog_ids[:train_idx])
        val_dog_ids = set(shuffled_dog_ids[train_idx:val_idx])
        test_dog_ids = set(shuffled_dog_ids[val_idx:])
        
        train_entries = []
        val_entries = []
        test_entries = []
        
        for dog_id, group in df.groupby("dog_id"):
            if dog_id in train_dog_ids:
                target_list = train_entries
            elif dog_id in val_dog_ids:
                target_list = val_entries
            elif dog_id in test_dog_ids:
                target_list = test_entries
            else:
                continue  # shouldn't happen
            
            for idx in group.index:
                row = group.loc[idx]
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
                entry = {
                    "path": str(audio_path.absolute()),
                    "audio_len": float(row["duration"]),
                    "label": dog_id_to_class[dog_id],
                    "breed": dog_id,  # Keep breed/dog_id for reference
                }
                target_list.append(entry)
        
        train_path = self._data_dir / "train_index.json"
        val_path = self._data_dir / "val_index.json"
        test_path = self._data_dir / "test_index.json"
        val_full_path = self._data_dir / "val_full_index.json"
        
        with open(train_path, "w") as f:
            json.dump(train_entries, f, indent=2)
        with open(val_path, "w") as f:
            json.dump(val_entries, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(test_entries, f, indent=2)
        # val_full combines val and test for backend training
        with open(val_full_path, "w") as f:
            json.dump(val_entries + test_entries, f, indent=2)
        
        print(f"Saved train index ({len(train_entries)} entries) to {train_path}")
        print(f"Saved val index ({len(val_entries)} entries) to {val_path}")
        print(f"Saved test index ({len(test_entries)} entries) to {test_path}")
        print(f"Saved val_full index ({len(val_entries) + len(test_entries)} entries) to {val_full_path}")

    def _build_test_index(self):
        """Build index for the test split."""
        test_dir = self._data_dir / "test"
        audio_dir = test_dir / "audio"
        metadata_path = test_dir / "metadata.csv"

        if not audio_dir.exists() or not metadata_path.exists():
            self._download_split("test", test_dir)
        df = pd.read_csv(metadata_path)
        index = []
        for _, row in tqdm(df.iterrows(), desc="Building test index", total=len(df)):
            audio_path = audio_dir / row["filename"]
            with sf.SoundFile(audio_path) as f:
                if f.frames == 0:
                    print(f"Warning: file {audio_path} has zero frames, skipping")
                    continue
            entry = {
                "path": str(audio_path.absolute()),
                "audio_len": float(row["duration"]),
            }
            index.append(entry)
        return index

    def _download_split(self, split: str, target_dir: Path):
        """
        Download a specific split ('train' or 'test') from Hugging Face using snapshot_download,
        copy all audio files (preserving original extensions) and create metadata.csv.
        For the train split, extracts dog ID from the parent folder name of the original file.
        """
        print(f"Downloading split '{split}' from Hugging Face via snapshot_download...")
        token = os.environ.get("HF_TOKEN", None)
        temp_dir = target_dir / "temp_download"
        
        snapshot_download(
            repo_id="ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset",
            repo_type="dataset",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=1,
            ignore_patterns=["*.parquet", "*.json", "*.csv", "*.txt", "*.md", "*.yaml"],
        )
        audio_dir = target_dir / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        metadata = []
        split_temp = temp_dir / split
        if split_temp.exists():
            search_dir = split_temp
        else:
            search_dir = temp_dir
        
        for audio_path in tqdm(sorted(search_dir.glob("**/*.wav")) + sorted(search_dir.glob("**/*.mp3")) + sorted(search_dir.glob("**/*.flac")), desc=f"Processing {split} audio"):
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
                    dog_id = int(audio_path.parent.name)
                    row["dog_id"] = dog_id
                metadata.append(row)
            except Exception as e:
                print(f"Warning: error processing {audio_path}: {e}, skipping")
                continue
        shutil.rmtree(temp_dir, ignore_errors=True)
        df = pd.DataFrame(metadata)
        csv_path = target_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(metadata)} entries to {csv_path}")