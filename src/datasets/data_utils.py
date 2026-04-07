from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
import pandas as pd
from typing import Optional
from pathlib import Path
import yadisk
import os
import json
from pathlib import Path
from typing import List, Dict
from pytubefix import YouTube
from pytubefix.cli import on_progress
import soundfile as sf
import numpy as np
import cv2

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # dataloaders init
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

#also everywhere extract yandex_token with os.get_env 

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

        return self

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

        if self.tracker_path.exists():
            with self.tracker_path.open() as f:
                self.data = json.load(f)
        return self

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


def _download_video(video_id: str, out_path: Path):
    url = f"https://www.youtube.com/watch?v={video_id}"

    yt = YouTube(url, on_progress_callback=on_progress)

    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    stream.download(output_path=str(out_path.parent), filename=out_path.name)


def _extract_audio_from_video(video_path: Path):
    """
    Extract audio using soundfile (works if backend supports codec).
    """
    waveform, sr = sf.read(video_path)
    return waveform, sr


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
        if bad_videos_path.exists():
            bad_videos = set(json.load(open(bad_videos_path)))
        else:
            bad_videos = set()

        if video_id in bad_videos:
            tracker.mark_skipped(video_id, reason="bad_video")
            return []
        missing = False
        for seg_idx, (start, end) in enumerate(segments):
            if not (video_dir / f"{seg_idx}_{start}_{end}.wav").exists():
                missing = True
                break
        waveform, sr = None, None

        if missing:
            try:
                if not video_path.exists():
                    _download_video(video_id, video_path)
                waveform, sr = _extract_audio_from_video(video_path)
            except Exception as e:
                error_msg = str(e).lower()
                is_bot = "bot" in error_msg or "captcha" in error_msg
                print(f"Error processing video {video_id}: {e}")
                if is_bot:
                    raise RuntimeError(
                        f"Bot detection encountered for video {video_id}. "
                        f"Use proxy / cookies / PO token."
                    )
                else:
                    if _is_temporary_error(e):
                        print(f"Temporary error for {video_id}, skipping.")
                    else:
                        bad_videos.add(video_id)
                        with open(bad_videos_path, "w") as f:
                            json.dump(list(bad_videos), f, indent=2)
                    tracker.mark_failed(video_id, reason=str(e), info={"temporary": _is_temporary_error(e)})
                return []

        outputs = []

        # ---- AUDIO PROCESSING ----
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

            # ---- FRAME EXTRACTION (cv2) ----
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

                        # save 1 FPS (you can adjust)
                        if frame_idx % int(fps) == 0:
                            frame_path = seg_frame_dir / f"frame_{saved_idx:04d}.jpg"
                            cv2.imwrite(str(frame_path), frame)
                            saved_idx += 1

                    frame_idx += 1

                cap.release()

            entry_out["frames_path"] = str(seg_frame_dir.absolute())

        outputs.append(entry_out)

    return outputs