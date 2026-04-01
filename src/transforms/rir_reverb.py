import random
from pathlib import Path
import logging
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
import os
import shutil

logger = logging.getLogger(__name__)


class RIRReverbAugment:
    """
    Add reverberation to raw audio samples using Room Impulse Responses (RIR) from Hugging Face.
    Downloads the dataset using snapshot_download and stores all .wav files in one folder.
    """

    def __init__(
        self,
        download_dir: str = "data/augmentation/rir",
        p: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.download_dir = Path(download_dir)
        self.p = p
        self.sample_rate = sample_rate

        self.download_dir.mkdir(parents=True, exist_ok=True)
        if not list(self.download_dir.glob("*.wav")):
            self._download_rirs()

        self._rir_files = []
        for wav_file in self.download_dir.glob("*.wav"):
            try:
                info = sf.info(wav_file)
                self._rir_files.append((wav_file, info.samplerate))
            except Exception as e:
                logger.warning(f"Could not read {wav_file}: {e}")

        logger.info(f"Loaded {len(self._rir_files)} RIR files from {self.download_dir}")

    def _download_rirs(self):
        """Download the RIRmega dataset using snapshot_download and flatten."""
        logger.info("Downloading RIRmega dataset from Hugging Face...")
        token = os.environ.get("HF_TOKEN", None)
        temp_dir = self.download_dir / "temp_download"
        snapshot_download(
            repo_id="mandipgoswami/rirmega",
            repo_type="dataset",
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=1,
            ignore_patterns=["*.parquet", "*.json", "*.csv", "*.txt", "*.md", "*.yaml"],
        )
        for wav_path in temp_dir.glob("**/*.wav"):
            shutil.move(str(wav_path), str(self.download_dir / wav_path.name))
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info(f"RIR files saved to {self.download_dir}")

    def _load_rir(self, file_path: Path, original_sr: int) -> torch.Tensor:
        """Load a single RIR file and resample if necessary."""
        audio, sr = sf.read(file_path, dtype='float32')
        rir = torch.from_numpy(audio).float()
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.sample_rate
            )
            rir = resampler(rir)
        if rir.dim() == 2:
            rir = rir.mean(dim=0)   
        elif rir.dim() > 2:
            rir = rir.squeeze()
        return rir

    def __call__(self, audio):
        if random.random() > self.p or not self._rir_files:
            return audio

        try:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float() if isinstance(audio, torch.Tensor) else torch.tensor(audio, dtype=torch.float32)

            file_path, orig_sr = random.choice(self._rir_files)
            rir = self._load_rir(file_path, orig_sr)
            rir = rir / (torch.max(torch.abs(rir)) + 1e-8)
            if audio_tensor.dim() == 1 and rir.dim() == 1:
                max_rir_len = 16000  
                if len(rir) > max_rir_len:
                    rir = rir[:max_rir_len]
                audio_np = audio_tensor.cpu().numpy()
                rir_np = rir.cpu().numpy()
                try:
                    from scipy import signal
                    audio_conv_np = signal.fftconvolve(audio_np, rir_np, mode='full')
                    audio_conv_np = audio_conv_np[:len(audio_np)]
                except ImportError:
                    audio_conv_np = np.convolve(audio_np, rir_np, mode='full')[:len(audio_np)]
                max_val = np.max(np.abs(audio_conv_np))
                if max_val > 1.0:
                    audio_conv_np = audio_conv_np / max_val

                if isinstance(audio, np.ndarray):
                    return audio_conv_np.astype(np.float32)
                else:
                    return torch.from_numpy(audio_conv_np).float()
            else:
                return audio
        except Exception as e:
            logger.warning(f"Error applying RIR reverberation: {e}, returning original audio")
            return audio