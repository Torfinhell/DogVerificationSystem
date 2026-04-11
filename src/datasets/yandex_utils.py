import yadisk
import os
from pathlib import Path
from typing import  Optional
import yadisk
import pandas as pd
import json
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

    def upload_chunk(self):
        if not self.buffer:
            return
        remote_path = f"/{self.file_csv}"
        if self.download_from_disk and self.yandex_token:
            with self.client:
                try:
                    self.file_csv.parent.mkdir(parents=True, exist_ok=True)
                    if self.client.exists(remote_path):
                        print(f"Downloading existing CSV from Yandex.Disk: {remote_path}")
                        self.client.download(remote_path, str(self.file_csv))
                except Exception as e:
                    print(f"Could not download existing CSV: {e}")
        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        df_chunk.to_csv(self.file_csv, mode="a", header=not self.file_csv.exists(), index=False)

        if self.yandex_token is not None:
            with self.client:
                remote_dir = Path(remote_path).parent
                if remote_dir != "/":
                    self.client.makedirs(remote_dir, exist_ok=True)
                self.client.upload(str(self.file_csv), remote_path, overwrite=True)
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
            remote_path = f"/{self.file_path}"
            with self.client:
                try:
                    self.file_path.parent.mkdir(parents=True, exist_ok=True)
                    if self.client.exists(remote_path) and self.download_from_disk:
                        print(f"Downloading {remote_path} from Yandex.Disk")
                        self.client.download(remote_path, str(self.file_path))
                    else:
                        print(f"File {remote_path} not found on Yandex.Disk, will create new")
                except yadisk.exceptions.PathNotFoundError:
                    print(f"File {remote_path} not found (404), will create new")
                except Exception as e:
                    print(f"Unexpected error: {e}, continuing without sync")
        return self

    def save(self):
        if self.yandex_token is None:
            return
        remote_path = f"/{self.file_path}"
        with self.client:
            remote_dir = Path(remote_path).parent
            if remote_dir != "/":
                self.client.makedirs(remote_dir, exist_ok=True)
            self.client.upload(str(self.file_path), remote_path, overwrite=True)

    def exists(self) -> bool:
        if self.file_path.exists():
            return True
        if self.yandex_token is None:
            return False
        remote_path = f"/{self.file_path}"
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
            remote_path = f"/{self.tracker_path}"  # full path with subdirs
            with self.client:
                try:
                    # Ensure local directory exists before download
                    self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
                    if self.client.exists(remote_path):
                        print(f"Downloading tracker {remote_path} from Yandex.Disk")
                        self.client.download(remote_path, str(self.tracker_path))
                    else:
                        print(f"Tracker {remote_path} not found on Yandex.Disk, starting fresh")
                except yadisk.exceptions.PathNotFoundError:
                    print(f"Tracker {remote_path} not found (404), starting fresh")
                except Exception as e:
                    print(f"Unexpected error accessing Yandex.Disk: {e}, continuing without sync")
        return self

    def save(self):
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tracker_path.open("w") as f:
            json.dump(self.data, f, indent=2)

        if self.yandex_token is not None:
            remote_path = f"/{self.tracker_path}"
            with self.client:
                # Ensure remote directory exists (create iteratively)
                remote_dir = Path(remote_path).parent
                if remote_dir != "/":
                    self.client.makedirs(remote_dir, exist_ok=True)
                self.client.upload(str(self.tracker_path), remote_path, overwrite=True)