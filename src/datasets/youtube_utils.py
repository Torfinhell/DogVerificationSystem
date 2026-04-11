import json
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from pytubefix import YouTube
from pytubefix import exceptions as pytubefix_exceptions
from pytubefix.cli import on_progress

from .yandex_utils import FILETracker


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
            raise pytubefix_exceptions.BotDetection(video_id) from e
        raise e


def _extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Extract audio from video using pydub (requires ffmpeg).
    Returns (waveform, sample_rate) where waveform is 1D numpy array.
    """
    audio = AudioSegment.from_file(str(video_path))
    audio = audio.set_channels(1)
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav = Path(tmp_file.name)
    audio.export(tmp_wav, format="wav")
    waveform, sr = sf.read(tmp_wav)
    tmp_wav.unlink()
    return waveform, sr


def youtube_download(
    entry: Dict,
    audio_root: Path = Path("./data/datasets/dog2vec_dataset/audio"),
    frames_root: Path = Path("./data/datasets/dog2vec_dataset/frames"),
    bad_videos_path: Path = Path("./data/datasets/dog2vec_dataset/bad_videos.json"),
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
            with open(bad_videos_path) as f:
                bad_videos = set(json.load(f))
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

        if not missing:
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
        waveform, sr = None, None
        try:
            if not video_path.exists():
                _download_video(video_id, video_path)
            waveform, sr = _extract_audio_from_video(video_path)
        except pytubefix_exceptions.BotDetection:
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