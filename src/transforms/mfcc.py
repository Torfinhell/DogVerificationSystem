import torch
import torch.nn as nn
import numpy as np
import librosa
from functools import partial

class MfccExtractor(nn.Module):
    """Extract MFCC features from a single audio waveform tensor."""

    def __init__(self, sample_rate: int, n_mfcc: int, hop_length_sec: float, win_length_sec: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.hop_length = int(sample_rate * hop_length_sec)
        self.win_length = int(sample_rate * win_length_sec)
        self.mfcc_extractor = partial(
            librosa.feature.mfcc,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.win_length   
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from a single audio waveform.

        Args:
            audio (torch.Tensor): 1D tensor of waveform samples.

        Returns:
            torch.Tensor: MFCC coefficients with shape (n_mfcc, time).
        """
        audio_np = audio.cpu().numpy()
        mfcc_np = self.mfcc_extractor(y=audio_np)   
        mfcc_tensor = torch.from_numpy(mfcc_np).float()
        return mfcc_tensor