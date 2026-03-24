import json
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH



class BarkopediaDataset(BaseDataset):
    """
    Barkopedia Individual Dog Recognition Dataset.

    Downloads from Hugging Face, saves audio files preserving their original format,
    and builds indexes (JSON) for train, validation, and test splits.
    The train and validation splits are created by splitting dog IDs from the
    original training set according to train_split.
    """

    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.9,
        data_dir: Path = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            part (str): partition name – "train", "val", or "test".
            train_split (float): fraction of dog IDs to use for training
                (the rest become validation). Ignored for test split.
            data_dir (Path, optional): root directory for the dataset.
                Default: ROOT_PATH / "data" / "datasets" / "barkopedia"
        """
        if part not in ["train", "val", "test"]:
            raise ValueError(f"Part must be 'train', 'val', or 'test', got '{part}'")

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "barkopedia"
        self._data_dir = Path(data_dir)
        self._train_split = train_split
        self._part = part

        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        """Load index from JSON if exists, otherwise create it."""
        index_path = self._data_dir / f"{self._part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        """
        Build index by scanning the local audio directory.
        For train/val, we use the training data and filter by dog IDs.
        For test, we use the test data directly.
        """
        if self._part == "test":
            return self._build_test_index()
        else:
            return self._build_train_val_index()

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

            if not audio_path.exists():
                print(f"Warning: file {audio_path} missing, skipping")
                continue
            if audio_path.stat().st_size == 0:
                print(f"Warning: empty file {audio_path}, skipping")
                continue
            try:
                with sf.SoundFile(audio_path) as f:
                    if f.frames == 0:
                        print(f"Warning: file {audio_path} has zero frames, skipping")
                        continue
            except Exception as e:
                print(f"Warning: corrupt file {audio_path}: {e}, skipping")
                continue

            entry = {
                "path": str(audio_path.absolute()),
                "audio_len": float(row["duration"]),
            }
            index.append(entry)
        return index

    def _build_train_val_index(self):
        """
        Build index for train or validation split.
        Uses the full training data (already downloaded) and filters
        by dog ID based on train_split.
        """
        train_dir = self._data_dir / "train"
        audio_dir = train_dir / "audio"
        metadata_path = train_dir / "metadata.csv"

        if not audio_dir.exists() or not metadata_path.exists():
            self._download_split("train", train_dir)

        df = pd.read_csv(metadata_path)

        if "dog_id" not in df.columns:
            raise RuntimeError(
                "The metadata.csv for the train split does not contain 'dog_id'. "
                "Please delete the 'train' folder in the dataset directory and rerun. "
                f"Path: {train_dir}"
            )

        dog_ids = sorted(df["dog_id"].unique())
        split_idx = int(len(dog_ids) * self._train_split)
        if self._part == "train":
            selected_dog_ids = set(dog_ids[:split_idx])
        else:
            selected_dog_ids = set(dog_ids[split_idx:])

        df_filtered = df[df["dog_id"].isin(selected_dog_ids)]

        index = []
        for _, row in tqdm(
            df_filtered.iterrows(),
            desc=f"Building {self._part} index",
            total=len(df_filtered),
        ):
            audio_path = audio_dir / row["filename"]

            # ---- File validation ----
            if not audio_path.exists():
                print(f"Warning: file {audio_path} missing, skipping")
                continue
            if audio_path.stat().st_size == 0:
                print(f"Warning: empty file {audio_path}, skipping")
                continue
            try:
                with sf.SoundFile(audio_path) as f:
                    if f.frames == 0:
                        print(f"Warning: file {audio_path} has zero frames, skipping")
                        continue
            except Exception as e:
                print(f"Warning: corrupt file {audio_path}: {e}, skipping")
                continue
            # -------------------------

            entry = {
                "path": str(audio_path.absolute()),
                "audio_len": float(row["duration"]),
                "label": int(row["dog_id"])-1,   # FIX: use proper mapping later
            }
            index.append(entry)
        return index

    def _download_split(self, split: str, target_dir: Path):
        """
        Download a specific split ('train' or 'test') from Hugging Face,
        copy all audio files (preserving original extensions) and create metadata.csv.
        For the train split, extracts dog ID from the parent folder name of the original file.
        """
        print(f"Downloading split '{split}' from Hugging Face...")
        dataset = load_dataset(
            "ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset",
            split=split,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(decode=False))

        audio_dir = target_dir / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)

        metadata = []
        for item in tqdm(dataset, desc=f"Saving {split} audio"):
            audio_info = item["audio"]
            src_path = audio_info["path"]


            audio_id = item.get("audio_id", Path(src_path).stem)
            original_ext = Path(src_path).suffix
            filename = f"{audio_id}{original_ext}"

            dst_path = audio_dir / filename
            shutil.copy2(src_path, dst_path)


            info = sf.info(src_path)
            duration = info.duration

            row = {
                "audio_id": audio_id,
                "filename": filename,
                "duration": duration,
            }
            if split == "train":
                dog_id = int(Path(src_path).parent.name)
                row["dog_id"] = dog_id

            metadata.append(row)

        df = pd.DataFrame(metadata)
        csv_path = target_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(metadata)} entries to {csv_path}")