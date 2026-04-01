import json
import random
import shutil
from pathlib import Path

import torch
import torchaudio
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


def _dog_id_to_class_map_from_metadata_df(df: pd.DataFrame) -> dict[int, int]:
    unique = sorted(int(x) for x in df["dog_id"].unique())
    return {did: i for i, did in enumerate(unique)}


def _mapping_payload(dog_id_to_class: dict[int, int]) -> dict:
    inv = {cls: did for did, cls in dog_id_to_class.items()}
    return {
        "num_classes": len(dog_id_to_class),
        "dog_id_to_class": {str(k): v for k, v in sorted(dog_id_to_class.items())},
        "class_to_dog_id": {str(k): v for k, v in sorted(inv.items())},
    }


def _write_mapping_files(data_dir: Path, dog_id_to_class: dict[int, int]) -> None:
    """Write canonical ``mapping.json`` and ``label_map.json`` (same content)."""
    payload = _mapping_payload(dog_id_to_class)
    for name in ("mapping.json", "label_map.json"):
        path = Path(data_dir) / name
        with path.open("w") as f:
            json.dump(payload, f, indent=2)


def _load_mapping_json(data_dir: Path) -> dict[int, int] | None:
    path = Path(data_dir) / "mapping.json"
    if not path.exists():
        return None
    with path.open() as f:
        raw = json.load(f)
    d = raw.get("dog_id_to_class", raw)
    return {int(k): int(v) for k, v in d.items()}


def read_num_classes_from_mapping(data_dir: Path | None = None) -> int:
    """
    Read ``num_classes`` from ``mapping.json`` (or ``label_map.json``) under the
    Barkopedia data directory. Use this to align Hydra ``num_classes`` / model
    ``n_class`` with the built indices.
    """
    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "barkopedia"
    data_dir = Path(data_dir)
    for name in ("mapping.json", "label_map.json"):
        path = data_dir / name
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            return int(data["num_classes"])
    raise FileNotFoundError(
        f"No mapping.json or label_map.json in {data_dir}. "
        "Build the dataset once (BarkopediaDataset will create them)."
    )


def _resolve_dog_id_to_class(df: pd.DataFrame, data_dir: Path) -> dict[int, int]:
    """
    Build dog_id -> class index for all dog_ids in ``df``.

    If ``mapping.json`` exists, reuse it and assign new consecutive indices only
    for dog_ids that are not yet in the map (e.g. new speakers after a dataset update).
    Otherwise enumerate sorted unique ``dog_id`` as 0..N-1.
    """
    unique_ids = sorted(int(x) for x in df["dog_id"].unique())
    existing = _load_mapping_json(data_dir)
    if existing is None:
        dog_id_to_class = _dog_id_to_class_map_from_metadata_df(df)
    else:
        dog_id_to_class = {int(k): int(v) for k, v in existing.items()}
        used_classes = set(dog_id_to_class.values())
        next_cls = max(used_classes) + 1 if used_classes else 0
        for did in unique_ids:
            if did not in dog_id_to_class:
                dog_id_to_class[did] = next_cls
                used_classes.add(next_cls)
                next_cls += 1
    _write_mapping_files(data_dir, dog_id_to_class)
    return dog_id_to_class


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
        train_split: float | None = 0.9,
        data_dir: Path = None,
        num_classes: int | None = None,
        random_seed: int = 42,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """
        Args:
            part (str): partition name – "train", "val", or "test".
            train_split (float | None): fraction of each dog_id's samples to use for training
                (the rest become validation). If None, all samples are used for training
                and validation is empty. Ignored for test split.
            data_dir (Path, optional): root directory for the dataset.
                Default: ROOT_PATH / "data" / "datasets" / "barkopedia"
            num_classes (int | None): if set (e.g. from Hydra), train/val splits
                assert this matches ``mapping.json`` and every index label lies in
                ``[0, num_classes)``.
            random_seed (int): random seed used when splitting per dog_id.
        """
        if part not in ["train", "val", "test"]:
            raise ValueError(f"Part must be 'train', 'val', or 'test', got '{part}'")

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "barkopedia"
        self._data_dir = Path(data_dir)
        self._train_split = train_split
        self._part = part
        self._random_seed = random_seed
        self.sample_rate = sample_rate

        index = self._get_or_load_index()
        self._assert_num_classes(index, num_classes)
        super().__init__(index, *args, **kwargs)

    def _assert_num_classes(self, index: list, num_classes: int | None) -> None:
        if self._part not in ("train", "val") or num_classes is None:
            return
        mapping_nc = read_num_classes_from_mapping(self._data_dir)
        assert mapping_nc == num_classes, (
            f"Barkopedia {self._data_dir}/mapping.json num_classes={mapping_nc} "
            f"does not match config num_classes={num_classes}"
        )
        for i, entry in enumerate(index):
            li = int(entry["label"])
            assert 0 <= li < num_classes, (
                f"{self._part}_index.json entry {i}: label {li} not in [0, {num_classes})"
            )

    def _get_or_load_index(self):
        """Load index from JSON if exists, otherwise create it."""
        index_path = self._data_dir / f"{self._part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            if self._part in ("train", "val"):
                self._build_train_val_indices()
            else:
                self._build_test_index()
            with index_path.open() as f:
                index = json.load(f)
        return index

    def _build_train_val_indices(self):
        """
        Build and save both train and validation indices.
        For each dog_id, its samples are split into train and val according to
        train_split, using a deterministic random seed. If train_split is None,
        all samples go to train and val is empty.
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
        dog_id_to_class = _resolve_dog_id_to_class(df, self._data_dir)
        train_entries = []
        val_entries = []
        rng = random.Random(self._random_seed)
        for dog_id, group in df.groupby("dog_id"):
            indices = list(group.index)
            rng.shuffle(indices)

            if self._train_split is None:
                split_idx = len(indices)
            else:
                split_idx = int(len(indices) * self._train_split)

            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            for idx_list, target_list in ((train_indices, train_entries),
                                          (val_indices, val_entries)):
                for idx in idx_list:
                    row = group.loc[idx]
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
                        "label": dog_id_to_class[dog_id],
                    }
                    target_list.append(entry)
        train_path = self._data_dir / "train_index.json"
        val_path = self._data_dir / "val_index.json"
        with open(train_path, "w") as f:
            json.dump(train_entries, f, indent=2)
        with open(val_path, "w") as f:
            json.dump(val_entries, f, indent=2)
        print(f"Saved train index ({len(train_entries)} entries) to {train_path}")
        print(f"Saved val index ({len(val_entries)} entries) to {val_path}")

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
                    try:
                        dog_id = int(audio_path.parent.name)
                        row["dog_id"] = dog_id
                    except (ValueError, IndexError):
                        print(f"Warning: could not extract dog_id from {audio_path}, skipping")
                        dst_path.unlink()  
                        continue
                
                metadata.append(row)
            except Exception as e:
                print(f"Warning: error processing {audio_path}: {e}, skipping")
                continue
        shutil.rmtree(temp_dir, ignore_errors=True)
        df = pd.DataFrame(metadata)
        csv_path = target_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(metadata)} entries to {csv_path}")
