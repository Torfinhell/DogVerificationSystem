import json
import random
from pathlib import Path
from typing import List, Dict, Union
from collections import defaultdict

import torch
import soundfile as sf
from torch.utils.data import Dataset

from .filter_dog2vec import (
    download_raw_jsons,
    download_info_youtube,
    final_filter_result,
)
from .base_dataset import BaseDataset


class Dog2VecDataset(BaseDataset):
    NAME = "Dog2Vec"

    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.7,
        max_videos_per_breed_context: int | None = None,
        data_dir: Union[str, Path] = None,
        random_seed: int = 42,
        *args,
        **kwargs,
    ):
        assert part in ["train", "val"], "part must be 'train' or 'val'"
        if data_dir is None:
            data_dir = Path("./data/dog2vec_dataset")
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.part = part
        self.random_seed = random_seed
        self.max_videos_per_breed_context = max_videos_per_breed_context
        self._ensure_pipeline_done()

        # Load or build index (follows Barkopedia pattern)
        index = self.load_index()
        super().__init__(index, *args, **kwargs)

    def load_index(self) -> List[Dict]:
        """Load index from JSON if exists, otherwise create it."""
        index_path = self.data_dir / f"{self.part}_index.json"
        if not index_path.exists():
            self.build_indices()
        with index_path.open() as f:
            return json.load(f)

    def build_indices(self):
        """Build the index for the selected part (train/val) and save to JSON."""
        final_json_path = self.data_dir / "filtered_index.json"
        if not final_json_path.exists():
            raise FileNotFoundError(f"Filtered index not found at {final_json_path}. Run pipeline first.")
        with open(final_json_path) as f:
            data = json.load(f)
        video_to_entries = defaultdict(list)
        for e in data:
            video_to_entries[e["video_id"]].append(e)

        # Create global label mapping for dog identities (video_id -> integer label)
        unique_videos = sorted(video_to_entries.keys())
        video_to_label = {video: i for i, video in enumerate(unique_videos)}

        # Create breed mapping (if needed)
        unique_breeds = sorted(set(e["label"] for e in data))
        breed_to_index = {b: i for i, b in enumerate(unique_breeds)}

        # Save mappings globally (like Barkopedia's mapping.json)
        mapping_path = self.data_dir / "mapping.json"
        if not mapping_path.exists():
            mapping = {
                "num_classes": len(video_to_label),
                "video_to_label": video_to_label,
                "breed_to_index": breed_to_index,
            }
            with open(mapping_path, "w") as f:
                json.dump(mapping, f, indent=2)

        # Build a list of all valid entries (filter out missing/corrupt files)
        all_entries = []
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
            all_entries.append({
                "path": str(path),
                "label": video_to_label[e["video_id"]],
                "breed": breed_to_index[e["breed"]],
                "video_id": e["video_id"],
                "audio_len": float(e["audio_len"]),
            })

        # Group by video_id again (after filtering)
        video_to_entries_final = defaultdict(list)
        for entry in all_entries:
            video_to_entries_final[entry["video_id"]].append(entry)

        # Split unique videos by train_split
        unique_videos_final = sorted(video_to_entries_final.keys())
        random.seed(self.random_seed)
        random.shuffle(unique_videos_final)

        split_idx = int(len(unique_videos_final) * self.train_split)
        if self.part == "train":
            selected_videos = set(unique_videos_final[:split_idx])
        else:  # part == "val"
            selected_videos = set(unique_videos_final[split_idx:])

        # Collect entries for the selected videos
        final_entries = [
            entry
            for video_id, entries in video_to_entries_final.items()
            if video_id in selected_videos
            for entry in entries
        ]

        # Shuffle the final entries (optional, but matches Barkopedia)
        random.shuffle(final_entries)

        # Save to part-specific index
        index_path = self.data_dir / f"{self.part}_index.json"
        with open(index_path, "w") as f:
            json.dump(final_entries, f, indent=2)
        print(f"Saved {self.part} index ({len(final_entries)} entries) to {index_path}")

    def _ensure_pipeline_done(self):
        """Run the preprocessing pipeline if the filtered index does not exist."""
        final_json_path = self.data_dir / "filtered_index.json"
        if not final_json_path.exists():
            print("Running Dog2Vec pipeline...")
            download_raw_jsons()
            download_info_youtube()
            final_filter_result(
                max_videos_per_breed_context=self.max_videos_per_breed_context
            )
