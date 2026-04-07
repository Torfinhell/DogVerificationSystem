import json
import random
from pathlib import Path
from typing import List, Dict, Union

import torch
import soundfile as sf
from torch.utils.data import Dataset
from collections import defaultdict
from .filter_dog2vec import (
    download_raw_jsons,
    download_info_youtube,
    final_filter_result,
)
from .base_dataset import BaseDataset


class Dog2VecDataset(BaseDataset):
    def __init__(
        self,
        part: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        max_videos_per_breed_context: int | None = None,
        data_dir: Union[str, Path] = None,
        random_seed: int = 42,
        *args,
        **kwargs,
    ):
        if part not in ["train", "val", "test", "val_full"]:
            raise ValueError(f"part must be 'train', 'val', 'test', or 'val_full', got {part}")

        # Validate splits sum to 1.0
        total_split = train_split + val_split + test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small float tolerance
            raise ValueError(f"train_split + val_split + test_split must equal 1.0, got {total_split}")

        self.part = part
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.max_videos_per_breed_context = max_videos_per_breed_context
        self.random_seed = random_seed
        self.data_dir = Path(data_dir or "./data/dog2vec_dataset")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.label_map_path = self.data_dir / "label_map.json"
        self.final_json_path = self.data_dir / "filtered_index.json"

        index = self._get_or_create_index()
        super().__init__(index, *args, **kwargs)

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
            final_filter_result(
                max_videos_per_breed_context=self.max_videos_per_breed_context
            )

        with open(self.final_json_path) as f:
            data = json.load(f)

        # For SV: each video_id represents a unique dog identity
        # Group by video_id to ensure all segments from same video have same label
        video_to_entries = defaultdict(list)
        for e in data:
            video_id = e["video_id"]
            video_to_entries[video_id].append(e)

        # Create unique IDs for each video (dog identity)
        unique_videos = sorted(video_to_entries.keys())
        video_to_label = {video: i for i, video in enumerate(unique_videos)}
        
        # Create breed mapping
        unique_breeds = sorted(set(e["label"] for e in data))
        breed_to_index = {b: i for i, b in enumerate(unique_breeds)}
        
        # Save label maps
        with open(self.label_map_path, "w") as f:
            label_maps = {
                "video_to_label": video_to_label,
                "breed_to_index": breed_to_index
            }
            json.dump(label_maps, f, indent=2)

        # Convert entries - each segment gets the label of its video
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
                    "label": video_to_label[e["video_id"]],  # Unique dog ID based on video
                    "breed": breed_to_index[e["label"]],  # Breed index
                    "video_id": e["video_id"],  # Keep video_id for reference
                    "audio_len": float(e["audio_len"]),
                }
            )

        # Group by video_id (unique dog identity)
        video_to_entries_final = defaultdict(list)
        for entry in entries:
            video_to_entries_final[entry["video_id"]].append(entry)

        # Split videos into train, val, and test (no overlap)
        unique_videos_final = sorted(video_to_entries_final.keys())
        random.seed(self.random_seed)
        random.shuffle(unique_videos_final)
        
        # Calculate split indices
        train_idx = int(len(unique_videos_final) * self.train_split)
        val_idx = train_idx + int(len(unique_videos_final) * self.val_split)
        
        train_videos = set(unique_videos_final[:train_idx])
        val_videos = set(unique_videos_final[train_idx:val_idx])
        test_videos = set(unique_videos_final[val_idx:])

        # Collect entries for the part
        if self.part == "train":
            selected_videos = train_videos
        elif self.part == "val":
            selected_videos = val_videos
        elif self.part == "test":
            selected_videos = test_videos
        elif self.part == "val_full":
            # Combined validation and test for backend training
            selected_videos = val_videos | test_videos
        
        selected_entries = [e for video_id, ents in video_to_entries_final.items() 
                          if video_id in selected_videos for e in ents]

        return selected_entries