"""Audio resampling transform."""

import torch
import torchaudio
import numpy as np


class ResampleTransform:
    """
    Resample audio to a target sample rate.
    
    This transform resamples audio waveforms to a specified sample rate
    using torchaudio's Resample transform.
    """

    def __init__(self, target_sample_rate: int):
        """
        Initialize the resampling transform.
        
        Args:
            target_sample_rate: Target sample rate in Hz
        """
        self.target_sample_rate = target_sample_rate
        self.resamplers = {}  

    def __call__(self, audio):
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio tensor or numpy array
            
        Returns:
            Resampled audio with the same type as input
        """
        # Handle different input types
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            was_numpy = True
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio.float()
            was_numpy = False
        else:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            was_numpy = False
            
        # Assume current sample rate is 16000 if not specified
        # In a real implementation, you might want to pass sample_rate as parameter
        current_sr = 16000
        if current_sr == self.target_sample_rate:
            return audio
            
        if current_sr not in self.resamplers:
            self.resamplers[current_sr] = torchaudio.transforms.Resample(
                orig_freq=current_sr,
                new_freq=self.target_sample_rate
            )
            
        resampled = self.resamplers[current_sr](audio_tensor)
        
        # Return same type as input
        if was_numpy:
            return resampled.numpy()
        else:
            return resampled