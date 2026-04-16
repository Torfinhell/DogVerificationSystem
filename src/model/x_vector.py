from torch import nn
import torch
class TDNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, dilation, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=num_groups,
            dilation=dilation,
            padding=padding,              
            padding_mode="reflect"
        )
        self.activation = activation
        self.norm=nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1) 
        return self.dropout(self.norm(self.activation(self.conv(x))))

class XVectorModel(nn.Module):
    def __init__(self, channel_sizes, output_dim, kernel_sizes, dilations, num_groups, activation=nn.ReLU(), dropout=0.1):
        assert len(channel_sizes)-1 == len(kernel_sizes) == len(dilations)
        super().__init__()
        self.tdnns = nn.ModuleList()
        for in_ch, out_ch, k, d in zip(channel_sizes[:-1], channel_sizes[1:], kernel_sizes, dilations):
            self.tdnns.append(
                TDNN(in_ch, out_ch, k, num_groups, d, activation, dropout)
            )
        self.segment_layers=nn.Sequential(
            nn.Linear(channel_sizes[-1]*2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


    def _masked_mean_std(self, x, spectral_feat_lengths, eps=1e-9):
        lengths=spectral_feat_lengths
        if lengths is None:
            mean = x.mean(dim=-1)
            std = torch.sqrt((x * x).mean(dim=-1) - mean * mean + eps)
            return mean, std

        lengths = lengths.to(x.device)
        if lengths.ndim == 1:
            lengths = lengths.unsqueeze(1)
        mask = torch.arange(x.size(-1), device=x.device).view(1, 1, -1) < lengths.view(-1, 1, 1)
        mask = mask.expand(-1, x.size(1), -1).to(x.dtype)
        masked = x * mask
        lengths = lengths.to(x.dtype).clamp_min(1).unsqueeze(1)
        mean = masked.sum(dim=-1) / lengths
        mean2 = (masked * masked).sum(dim=-1) / lengths
        std = torch.sqrt((mean2 - mean * mean).clamp_min(eps))
        return mean, std

    def forward(self, spectral_feat, spectral_feat_lengths):
        x = spectral_feat
        for layer in self.tdnns:
            x = layer(x)
        mean, std = self._masked_mean_std(x, spectral_feat_lengths)
        x = torch.cat([mean, std], dim=1)
        embedding = self.segment_layers[:-1](x)
        logits = self.segment_layers[-1](embedding)
        return {"logits": logits, "embedding": embedding}