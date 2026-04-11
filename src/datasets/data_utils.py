import json
import os
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import yadisk
from hydra.utils import instantiate
from pytubefix import YouTube
from pytubefix.cli import on_progress

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
from pytubefix import exceptions as pytubefix_exceptions
import pytubefix


def _non_epoch_metric_names(metric_list):
    from src.metrics.epoch_metric import EpochMetric

    return [m.name for m in metric_list if not isinstance(m, EpochMetric)]


def _build_backend_keys(backends):
    backend_name_counts = {}
    backend_keys = []
    for backend in backends:
        base_name = backend.__class__.__name__
        count = backend_name_counts.get(base_name, 0) + 1
        backend_name_counts[base_name] = count
        backend_keys.append(base_name if count == 1 else f"{base_name}_{count}")
    return backend_keys


def metric_keys_for_partition(metrics, part_name):
    """Input: metrics dict, partition name. Output: non-epoch metric names."""
    from src.metrics.epoch_metric import EpochMetric

    partition_metrics = metrics.get(part_name, [])
    if isinstance(partition_metrics, dict):
        partition_metrics = partition_metrics.get("main", [])
    return [m.name for m in partition_metrics if not isinstance(m, EpochMetric)]


def _set_num_classes_on_metric_cfg(metric_cfg, num_classes):
    if not hasattr(metric_cfg, "get"):
        return
    if metric_cfg.get("_target_") == "src.metrics.ClassificationMetric":
        classification_metric_cfg = metric_cfg.get("classification_metric")
        if classification_metric_cfg is not None and hasattr(classification_metric_cfg, "__setitem__"):
            classification_metric_cfg["num_classes"] = num_classes
    if "num_classes" in metric_cfg:
        metric_cfg["num_classes"] = num_classes


def get_metrics_and_backends(config, dataloaders):
    """Input: config and dataloaders. Output: instantiated metrics and backends."""
    metrics_cfg = deepcopy(config.metrics)

    for part in ["val", "test"]:
        if part in dataloaders and part in metrics_cfg:
            num_classes = dataloaders[part].dataset.num_classes
            for met_cfg in metrics_cfg[part]:
                _set_num_classes_on_metric_cfg(met_cfg, num_classes)

    metrics = instantiate(metrics_cfg)
    backends = []
    if config.get("backends") is not None:
        backends = instantiate(config.backends)
    backend_keys = _build_backend_keys(backends)
    train_metric_objects = metrics.get("train", [])
    val_metric_objects = metrics.get("val", [])
    test_metric_objects = metrics.get("test", [])
    backend_metric_objects = {}
    backend_metric_names = {}
    if backends and isinstance(test_metric_objects, list):
        for backend_key in backend_keys:
            per_backend_metrics = deepcopy(test_metric_objects)
            backend_metric_objects[backend_key] = per_backend_metrics
            backend_metric_names[backend_key] = _non_epoch_metric_names(per_backend_metrics)
    prepared_metrics = {
        "train": train_metric_objects,
        "val": val_metric_objects,
        "test": {"backends": backend_metric_objects},
        "metric_keys": {
            "train": _non_epoch_metric_names(train_metric_objects),
            "val": _non_epoch_metric_names(val_metric_objects),
            "test_backends": backend_metric_names,
        },
        "backend_keys": backend_keys,
    }
    if "inference" in metrics:
        prepared_metrics["inference"] = metrics["inference"]
        prepared_metrics["metric_keys"]["inference"] = _non_epoch_metric_names(metrics["inference"])
    return prepared_metrics, backends

def inf_loop(dataloader):
    """Input: finite dataloader. Output: endless dataloader iterator."""
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """Input: transform mapping and device. Output: transforms moved to device."""
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """Input: config and device. Output: dataloaders and batch transforms."""
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)
    datasets = instantiate(config.datasets)
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.dataloader_standard.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.dataloader_standard.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        dataloader_kwargs = {
            "dataset": dataset,
            "collate_fn": collate_fn,
            "drop_last": (dataset_partition == "train"),
            "worker_init_fn": set_worker_seed,
        }
        dataloader_kwargs["shuffle"] = (dataset_partition == "train")
        partition_dataloader = instantiate(config.dataloader.dataloader_standard, **dataloader_kwargs)
        dataloaders[dataset_partition] = partition_dataloader
    return dataloaders, batch_transforms

class CsvChunkDownloader:
    def __init__(
        self,
        file_csv,
        columns: list[str],
        chunk_rows: Optional[int] = 100,
        download_from_disk=False,
    ):
        self.file_csv = Path(file_csv)
        self.columns = columns
        self.buffer = []
        self.chunk_rows = chunk_rows
        self.yandex_token = os.getenv("YANDEX_TOKEN")
        self.download_from_disk = download_from_disk
        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def __enter__(self):
        return self

    def update_csv(self, new_row: pd.Series):
        self.buffer.append(new_row.to_dict())
        if self.chunk_rows is not None and len(self.buffer) >= self.chunk_rows:
            self.upload_chunk()

    def upload_chunk(self):
        if not self.buffer:
            return
        remote_path = f"/{self.file_csv.name}"
        if self.download_from_disk and self.client.exists(remote_path):
            with self.client:
                print(f"Downloading existing CSV from Yandex.Disk: {remote_path}")
                self.client.download(remote_path, str(self.file_csv))
        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        df_chunk.to_csv(
            self.file_csv,
            mode="a",
            header=not self.file_csv.exists(),
            index=False,
        )

        if self.yandex_token is not None:
            with self.client:
                self.client.upload(
                    str(self.file_csv), f"/{self.file_csv.name}", overwrite=True
                )

        self.buffer.clear()

    def get_csv(self, default_columns):
        if not self.file_csv.exists():
            return pd.DataFrame(columns=default_columns)
        return pd.read_csv(self.file_csv)

    def __exit__(self, exc_type, exc_value, traceback):
        self.upload_chunk()
        return False
    
class FILEDownloader:
    def __init__(
        self,
        file_path: str | Path,
        download_from_disk: bool = True,
    ):
        self.file_path = Path(file_path)
        self.yandex_token = os.getenv("YANDEX_TOKEN")
        self.download_from_disk = download_from_disk

        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def __enter__(self):
        if self.yandex_token is not None:
            remote_path = f"/{self.file_path.name}"
            with self.client:
                if self.client.exists(remote_path) and self.download_from_disk:
                    print(f"Downloading {remote_path} from Yandex.Disk")
                    self.client.download(remote_path, str(self.file_path))
                else:
                    print(f"File {remote_path} not found on Yandex.Disk, will create new")

    def save(self):
        if self.yandex_token is None:
            return

        remote_path = f"/{self.file_path.name}"

        with self.client:
            print(f"Uploading {remote_path} to Yandex.Disk")
            self.client.upload(
                str(self.file_path),
                remote_path,
                overwrite=True,
            )

    def exists(self) -> bool:
        if self.file_path.exists():
            return True

        if self.yandex_token is None:
            return False

        remote_path = f"/{self.file_path.name}"
        with self.client:
            return self.client.exists(remote_path)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.save()
        return False


class FILETracker:
    def __init__(self, tracker_path: str | Path, download_from_disk: bool = True):
        self.tracker_path = Path(tracker_path)
        self.download_from_disk = download_from_disk
        self.yandex_token = os.getenv("YANDEX_TOKEN")
        self.data = {
            "in_progress": {},
            "completed": {},
            "failed": {},
            "skipped": {},
        }
        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def __enter__(self):
        if self.yandex_token is not None and self.download_from_disk:
            remote_path = f"/{self.tracker_path.name}"
            with self.client:
                if self.client.exists(remote_path):
                    print(f"Downloading tracker {remote_path} from Yandex.Disk")
                    self.client.download(str(self.tracker_path), str(self.tracker_path))
                else:
                    print(f"Tracker {remote_path} not found on Yandex.Disk, starting fresh")

    def save(self):
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tracker_path.open("w") as f:
            json.dump(self.data, f, indent=2)

        if self.yandex_token is not None:
            remote_path = f"/{self.tracker_path.name}"
            with self.client:
                self.client.upload(str(self.tracker_path), remote_path, overwrite=True)

    def mark_started(self, video_id: str, info: dict | None = None) -> None:
        self.data["in_progress"][video_id] = {
            "status": "in_progress",
            "info": info or {},
        }

    def mark_done(self, video_id: str, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["completed"][video_id] = {
            "status": "completed",
            "info": info or {},
        }

    def mark_failed(self, video_id: str, reason: str | None = None, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["failed"][video_id] = {
            "status": "failed",
            "reason": reason,
            "info": info or {},
        }

    def mark_skipped(self, video_id: str, reason: str | None = None, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["skipped"][video_id] = {
            "status": "skipped",
            "reason": reason,
            "info": info or {},
        }

    def get_status(self, video_id: str) -> str:
        if video_id in self.data["completed"]:
            return "completed"
        if video_id in self.data["failed"]:
            return "failed"
        if video_id in self.data["skipped"]:
            return "skipped"
        if video_id in self.data["in_progress"]:
            return "in_progress"
        return "unknown"

    def summary(self) -> dict:
        return {key: len(value) for key, value in self.data.items()}

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()
        return False


def _is_temporary_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in [
        "timeout",
        "temporarily unavailable",
        "connection",
        "429",
        "too many requests",
        "internal error",
    ])


def _is_bot_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ["bot", "captcha", "detected as a bot"])


def _download_video(video_id: str, out_path: Path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        if stream is None:
            raise RuntimeError(f"No progressive stream found for {video_id}")
        stream.download(output_path=str(out_path.parent), filename=out_path.name)
    except Exception as e:
        if _is_bot_error(e):
            raise pytubefix.exceptions.BotDetection(video_id) from e
        raise e


def youtube_download(
    entry: Dict,
    audio_root: Path = Path("./data/dog2vec_dataset/audio"),
    frames_root: Path = Path("./data/dog2vec_dataset/frames"),
    bad_videos_path: Path = Path("./data/dog2vec_dataset/bad_videos.json"),
    audio_only: bool = True,
) -> List[Dict]:
    video_id = entry["video_id"]
    breed = entry["breed"]
    segments = entry["segments"]

    video_dir = audio_root / breed / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{video_id}.mp4"

    if not audio_only:
        frame_dir = frames_root / breed / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

    tracker_path = audio_root.parent / "download_tracker.json"
    with FILETracker(tracker_path) as tracker:
        current_status = tracker.get_status(video_id)
        if current_status == "completed":
            return []
        tracker.mark_started(video_id, {"breed": breed, "segments": segments})

        # Check bad videos list
        if bad_videos_path.exists():
            with open(bad_videos_path) as f:
                bad_videos = set(json.load(f))
        else:
            bad_videos = set()

        if video_id in bad_videos:
            tracker.mark_skipped(video_id, reason="bad_video")
            return []

        # Check if all segments already exist
        missing = False
        for seg_idx, (start, end) in enumerate(segments):
            if not (video_dir / f"{seg_idx}_{start}_{end}.wav").exists():
                missing = True
                break

        if not missing:
            # All segments already present
            outputs = []
            for seg_idx, (start, end) in enumerate(segments):
                audio_path = video_dir / f"{seg_idx}_{start}_{end}.wav"
                entry_out = {
                    "path": str(audio_path.absolute()),
                    "label": breed,
                    "audio_len": float(end - start),
                }
                if not audio_only:
                    seg_frame_dir = frame_dir / f"{seg_idx}_{start}_{end}"
                    entry_out["frames_path"] = str(seg_frame_dir.absolute())
                outputs.append(entry_out)
            tracker.mark_done(video_id, {"breed": breed, "segments_count": len(segments)})
            return outputs

        # Need to download and process
        waveform, sr = None, None
        try:
            if not video_path.exists():
                _download_video(video_id, video_path)
            waveform, sr = _extract_audio_from_video(video_path)
        except pytubefix.exceptions.BotDetection:
            print(f"Bot detection for {video_id}, skipping.")
            tracker.mark_failed(video_id, reason="bot_detection")
            return []
        except Exception as e:
            error_msg = str(e).lower()
            if _is_bot_error(e):
                print(f"Bot detection for {video_id}, skipping.")
                tracker.mark_failed(video_id, reason="bot_detection")
                return []
            print(f"Error processing video {video_id}: {e}")
            if _is_temporary_error(e):
                print(f"Temporary error for {video_id}, skipping.")
                tracker.mark_failed(video_id, reason=str(e), info={"temporary": True})
            else:
                bad_videos.add(video_id)
                with open(bad_videos_path, "w") as f:
                    json.dump(list(bad_videos), f, indent=2)
                tracker.mark_failed(video_id, reason=str(e), info={"temporary": False})
            return []

        outputs = []
        for seg_idx, (start, end) in enumerate(segments):
            duration = end - start
            audio_path = video_dir / f"{seg_idx}_{start}_{end}.wav"

            if not audio_path.exists():
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment = waveform[start_sample:end_sample]
                if segment.ndim > 1:
                    segment = np.mean(segment, axis=1)
                sf.write(audio_path, segment, sr)

            entry_out = {
                "path": str(audio_path.absolute()),
                "label": breed,
                "audio_len": float(duration),
            }

            if not audio_only:
                seg_frame_dir = frame_dir / f"{seg_idx}_{start}_{end}"
                seg_frame_dir.mkdir(parents=True, exist_ok=True)
                if not any(seg_frame_dir.iterdir()):
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    start_frame = int(start * fps)
                    end_frame = int(end * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    frame_idx = start_frame
                    saved_idx = 0
                    while frame_idx < end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % int(fps) == 0:
                            frame_path = seg_frame_dir / f"frame_{saved_idx:04d}.jpg"
                            cv2.imwrite(str(frame_path), frame)
                            saved_idx += 1
                        frame_idx += 1
                    cap.release()
                entry_out["frames_path"] = str(seg_frame_dir.absolute())

            outputs.append(entry_out)

        tracker.mark_done(video_id, {"breed": breed, "segments_count": len(segments)})
        return outputs