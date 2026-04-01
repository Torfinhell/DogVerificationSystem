import random
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
import os

logger = logging.getLogger(__name__)


class MusanNoiseAugment:
    """
    Add noise from MUSAN dataset to raw audio samples.
    Downloads the dataset using snapshot_download and stores all .wav files in one folder.
    """

    def __init__(
        self,
        download_dir: str = "data/augmentation/musan",
        noise_types: Optional[List[str]] = None,
        snr_range: Tuple[float, float] = (10, 20),
        p: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.download_dir = Path(download_dir)
        self.noise_types = noise_types or ["noise"]
        self.snr_range = snr_range
        self.p = p
        self.sample_rate = sample_rate
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if not list(self.download_dir.glob("*.wav")):
            self._download_musan()

        self._audio_files = []
        for f in self.download_dir.glob("*.wav"):
            try:
                info = sf.info(f)
                self._audio_files.append((f, info.samplerate))
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")

        logger.info(f"Loaded {len(self._audio_files)} noise files from {self.download_dir}")

    def _download_musan(self):
        """Download the MUSAN dataset using snapshot_download and flatten."""
        logger.info("Downloading MUSAN dataset from Hugging Face...")
        token = os.environ.get("HF_TOKEN", None)
        temp_dir = self.download_dir / "temp_download"
        snapshot_download(
            repo_id="bilguun/musan-noise",
            repo_type="dataset",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=1,
        )
        for wav_path in temp_dir.glob("**/*.wav"):
            shutil.move(str(wav_path), str(self.download_dir / wav_path.name))
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info(f"MUSAN files saved to {self.download_dir}")
    def _load_noise(self, file_path: Path, original_sr: int) -> torch.Tensor:
        audio, sr = sf.read(file_path, dtype='float32')
        audio_tensor = torch.from_numpy(audio).float()
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.sample_rate
            )
            audio_tensor = resampler(audio_tensor)
        return audio_tensor

    def _adjust_noise_level(self, signal: torch.Tensor, noise: torch.Tensor, target_snr: float) -> torch.Tensor:
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        target_noise_power = signal_power / (10 ** (target_snr / 10))
        adjustment_factor = torch.sqrt(target_noise_power / (noise_power + 1e-8))
        return noise * adjustment_factor

    def __call__(self, audio):
        if random.random() > self.p or len(self._audio_files) == 0:
            return audio
        try:
            file_path, orig_sr = random.choice(self._audio_files)
            noise = self._load_noise(file_path, orig_sr)
            if len(noise) < len(audio):
                num_repeats = (len(audio) // len(noise)) + 1
                noise = noise.repeat(num_repeats)
            max_start = len(noise) - len(audio)
            start_idx = random.randint(0, max(0, max_start))
            noise = noise[start_idx : start_idx + len(audio)]
            target_snr = random.uniform(self.snr_range[0], self.snr_range[1])
            noise = self._adjust_noise_level(audio, noise, target_snr)
            augmented = audio + noise
            if isinstance(audio, np.ndarray):
                return augmented.numpy()
            return augmented
        except Exception as e:
            logger.warning(f"Error applying MUSAN noise augmentation: {e}, returning original audio")
            return audio