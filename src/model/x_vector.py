from torch import nn
import torch
class TDNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, dilation, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=num_groups,
            dilation=dilation,
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
    def forward(self, extracted_feature, **batch):
        x = extracted_feature
        for layer in self.tdnns:
            x = layer(x)
        x = torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=1)
        embedding = self.segment_layers[:-1](x)
        logits = self.segment_layers[-1](embedding)
        return {"logits": logits, "embedding": embedding}