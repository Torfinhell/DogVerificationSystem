import json
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import yadisk


def _upload_with_retry(client, local_path, remote_path, max_wait_seconds=300):
    """
    Upload a file with retry on ResourceIsLockedError.
    Uses exponential backoff: 1, 2, 4, 8, 16, 32 seconds (max 32).
    Total wait up to max_wait_seconds.
    """
    start_time = time.time()
    delay = 1
    while True:
        try:
            client.upload(str(local_path), remote_path, overwrite=True)
            return  # success
        except yadisk.exceptions.ResourceIsLockedError:
            elapsed = time.time() - start_time
            if elapsed + delay > max_wait_seconds:
                raise  # timeout exceeded, re-raise the last error
            print(f"Resource locked, retrying in {delay}s... (total wait {elapsed:.1f}s)")
            time.sleep(delay)
            delay = min(delay * 2, 32)  # exponential backoff, cap at 32s
        except Exception:
            # Other errors are not retried
            raise


class CsvChunkDownloader:
    def __init__(self, file_csv, columns: list[str], chunk_rows: Optional[int] = 100, download_from_disk=False):
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

    def update_csv(self, row: pd.Series):
        self.buffer.append(row.to_dict())
        if self.chunk_rows is not None and len(self.buffer) >= self.chunk_rows:
            self.upload_chunk()

    def upload_chunk(self):
        if not self.buffer:
            return
        remote_path = f"/{self.file_csv.as_posix()}"
        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        df_chunk.to_csv(self.file_csv, mode="a", header=not self.file_csv.exists(), index=False)

        if self.yandex_token is not None:
            with self.client:
                remote_dir = Path(remote_path).parent
                if remote_dir.as_posix() != "/" and not self.client.exists(remote_dir.as_posix()):
                    self.client.makedirs(remote_dir.as_posix())
                _upload_with_retry(self.client, self.file_csv, remote_path)
        self.buffer.clear()

    def get_csv(self, default_columns):
        if not self.file_csv.exists():
            return pd.DataFrame(columns=default_columns)
        return pd.read_csv(self.file_csv)

    def __exit__(self, exc_type, exc_value, traceback):
        self.upload_chunk()
        return False


class FILEDownloader:
    def __init__(self, file_path: str | Path, download_from_disk: bool = True):
        self.file_path = Path(file_path)
        self.yandex_token = os.getenv("YANDEX_TOKEN")
        self.download_from_disk = download_from_disk
        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def __enter__(self):
        if self.yandex_token is not None:
            remote_path = f"/{self.file_path.as_posix()}"
            with self.client:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                if self.client.exists(remote_path) and self.download_from_disk:
                    print(f"Downloading {remote_path} from Yandex.Disk")
                    self.client.download(remote_path, str(self.file_path))
                else:
                    print(f"File {remote_path} not found on Yandex.Disk, will create new")
        return self

    def save(self):
        if self.yandex_token is None:
            return
        remote_path = f"/{self.file_path.as_posix()}"
        with self.client:
            remote_dir = Path(remote_path).parent
            if remote_dir.as_posix() != "/" and not self.client.exists(remote_dir.as_posix()):
                self.client.makedirs(remote_dir.as_posix())
            _upload_with_retry(self.client, self.file_path, remote_path)

    def exists(self) -> bool:
        if self.file_path.exists():
            return True
        if self.yandex_token is None:
            return False
        remote_path = f"/{self.file_path.as_posix()}"
        with self.client:
            return self.client.exists(remote_path)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.save()
        return False


class FILETracker:
    def __init__(self, tracker_path: str | Path, download_from_disk: bool = True, auto_sync_interval: int = 0):
        self.tracker_path = Path(tracker_path)
        self.download_from_disk = download_from_disk
        self.yandex_token = os.getenv("YANDEX_TOKEN")
        self.auto_sync_interval = auto_sync_interval
        self._dirty = False
        self._mod_count = 0
        self.data = {
            "in_progress": {},
            "completed": {},
            "failed": {},
            "skipped": {},
            "version": 0,
        }
        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def _download(self):
        if self.yandex_token is None or not self.download_from_disk:
            return
        remote_path = f"/{self.tracker_path.as_posix()}"
        with self.client:
            if self.client.exists(remote_path):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                self.client.download(remote_path, str(tmp_path))
                with open(tmp_path) as f:
                    remote_data = json.load(f)
                tmp_path.unlink()
                if remote_data.get("version", 0) > self.data.get("version", 0):
                    self.data = remote_data

    def _upload(self, force=False):
        if self.yandex_token is None:
            return
        if not self._dirty and not force:
            return
        self.data["version"] = self.data.get("version", 0) + 1
        remote_path = f"/{self.tracker_path.as_posix()}"
        with self.client:
            remote_dir = Path(remote_path).parent
            if remote_dir.as_posix() != "/" and not self.client.exists(remote_dir.as_posix()):
                self.client.makedirs(remote_dir.as_posix())
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                json.dump(self.data, tmp, indent=2)
                tmp_path = Path(tmp.name)
            _upload_with_retry(self.client, tmp_path, remote_path)
            tmp_path.unlink()
        self._dirty = False
        self._mod_count = 0

    def __enter__(self):
        self._download()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._upload(force=True)
        return False

    def _maybe_auto_sync(self):
        if self.auto_sync_interval > 0:
            self._mod_count += 1
            if self._mod_count >= self.auto_sync_interval:
                self._upload()

    def mark_started(self, video_id: str, info: dict | None = None) -> None:
        self.data["in_progress"][video_id] = {"status": "in_progress", "info": info or {}}
        self._dirty = True
        self._maybe_auto_sync()

    def mark_done(self, video_id: str, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["completed"][video_id] = {"status": "completed", "info": info or {}}
        self._dirty = True
        self._maybe_auto_sync()

    def mark_failed(self, video_id: str, reason: str | None = None, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["failed"][video_id] = {"status": "failed", "reason": reason, "info": info or {}}
        self._dirty = True
        self._maybe_auto_sync()

    def mark_skipped(self, video_id: str, reason: str | None = None, info: dict | None = None) -> None:
        self.data["in_progress"].pop(video_id, None)
        self.data["skipped"][video_id] = {"status": "skipped", "reason": reason, "info": info or {}}
        self._dirty = True
        self._maybe_auto_sync()

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
        return {key: len(value) for key, value in self.data.items() if key != "version"}

    def refresh(self):
        self._download()

    def flush(self):
        self._upload(force=True)