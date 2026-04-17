import torch
import torch.nn as nn
import random
import logging

logger = logging.getLogger(__name__)


class FbankAug(nn.Module):
    """
    Frequency and time masking augmentation for filterbank features.
    Applies random frequency and time masks to spectrogram-like inputs.
    """

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        """
        Apply frequency and time masking to filterbank features.

        Args:
            x: Input tensor of shape (batch, freq, time) or (freq, time)

        Returns:
            Augmented tensor with same shape as input
        """
        try:
            if x.dim() == 2:
                x = x.unsqueeze(0)
                was_single = True
            else:
                was_single = False

            if x.dim() != 3:
                logger.warning(f"Expected 2D or 3D input, got {x.dim()}D. Skipping augmentation.")
                return x
            x = self.mask_along_axis(x, dim=1)
            x = self.mask_along_axis(x, dim=2)
            if was_single:
                x = x.squeeze(0)

            return x

        except Exception as e:
            logger.warning(f"Error applying FbankAug: {e}, returning original input")
            return x.squeeze()