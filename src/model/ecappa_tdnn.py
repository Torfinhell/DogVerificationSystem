from torch import nn
from torch.nn import Sequential
from einops import rearrange
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
class Res2Block(nn.Module):
    def __init__(self, input_channels, kernel_size, dilation, num_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottle_neck=nn.Conv1d(input_channels, 1, kernel_size=1)
        self.convs_bns=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1,1, kernel_size, dilation=dilation, padding=(dilation * (kernel_size - 1)) // 2),
                nn.ReLU(),
                nn.BatchNorm1d(1)
            )
            for _ in range(num_groups)
        ]
        )
        self.output_conv= nn.Sequential(
            nn.Conv1d(1, input_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(input_channels)
        )
        self.num_groups=num_groups
    def forward(self, x):
        x = self.bottle_neck(x)
        outs = []
        truncated_x_shape=x.shape[-1]-x.shape[-1]%4
        for i, chunk in enumerate(torch.split(x[..., :truncated_x_shape], x.shape[-1]//self.num_groups, dim=-1)):
            if outs:
                input_tensor = outs[-1] + chunk 
            else:
                input_tensor = chunk
                
            out = self.convs_bns[i](input_tensor)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        return self.output_conv(x)

class SeBlock(nn.Module):
    def __init__(self, input_channels, reduction_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net=nn.Sequential(
            nn.Linear(input_channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, input_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.mean(-1))[..., None]* x
class SeRes2Block(nn.Module):
    def __init__(self, input_channels, reduction_dim, k, d, num_groups,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1=nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(input_channels)
        )
        self.block2=nn.Sequential(
            Res2Block(input_channels, kernel_size=k, dilation=d, num_groups=num_groups),
            nn.ReLU(),
            nn.BatchNorm1d(input_channels)
        )
        self.block3=nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=1)
        )
        self.se_block=SeBlock(input_channels, reduction_dim)
    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        return self.se_block(x)
class AttnStatPool(nn.Module):
    def __init__(self,input_channels, reduction_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net=nn.Sequential(
            nn.Linear(input_channels,reduction_dim, bias=True),
            nn.ReLU(),
            nn.Linear(reduction_dim, input_channels, bias=True)
        )
    def forward(self, x):
        normalized_weights=F.softmax(self.net(rearrange(x, "b c t -> b t c")), dim=-1)
        normalized_weights=rearrange(normalized_weights, "b t c -> b c t")
        mean=(normalized_weights*x).mean(dim=-1)
        std=(normalized_weights*(x**2)-(mean**2)[..., None]).sum(dim=-1)
        return torch.cat([mean, std], dim=1)
class PreEmphasis(torch.nn.Module):
    def __init__(self, num_feats, alpha: float = 0.97):
        super().__init__()
        self.alpha = alpha
        self.num_feats = num_feats
        filter_data = torch.FloatTensor([-self.alpha, 1.]).view(1, 1, 2).repeat(num_feats, 1, 1)
        self.register_buffer('flipped_filter', filter_data)

    def forward(self, x):
        x = x.unsqueeze(2) 
        x = F.pad(x, (1, 0, 0, 0), mode='reflect')
        x = x.squeeze(2)
        return F.conv1d(x, self.flipped_filter, groups=self.num_feats)

        
class EcappaTDNN(nn.Module):
    """
    Factory class that returns either pretrained or custom ECAPPA-TDNN implementation.
    """
    def __new__(cls, n_feats, input_channels, reduction_dim, output_dim, dropout, hidden_channels, num_groups, pre_emphasis_alpha, use_pre_emphasis=True, pretrained=False):
        """
        Args:
            n_feats (int): number of input features.
            output_dim (int): number of output features.
            input_channels (int): number of channels for first conv.
            reduction_dim (int): reduction dimension for SE blocks.
            dropout (float): dropout rate.
            hidden_channels (int): number of hidden channels.
            num_groups (int): number of groups in Res2Block.
            pre_emphasis_alpha (float): alpha for pre-emphasis filter.
            use_pre_emphasis (bool): whether to use pre-emphasis (only for custom implementation).
            pretrained (bool): if True, load pretrained ECAPPA-TDNN from SpeechBrain; if False, use custom implementation.
        """
        if pretrained:
            return EcappaTDNNPretrained()
        else:
            return EcappaTDNNCustom(n_feats, input_channels, reduction_dim, output_dim, dropout, hidden_channels, num_groups, pre_emphasis_alpha, use_pre_emphasis)


class EcappaTDNNCustom(nn.Module):
    """
    Custom ECAPPA-TDNN implementation.
    Forward takes spectral features as input.
    """
    def __init__(self, n_feats, input_channels, reduction_dim, output_dim, dropout, hidden_channels, num_groups, pre_emphasis_alpha, use_pre_emphasis=True):
        """Initialize custom ECAPPA-TDNN implementation."""
        super().__init__()
        assert input_channels%num_groups==0, "This should be divisable for res2 blocks"
        self.use_pre_emphasis = use_pre_emphasis
        self.pre_emphasis = PreEmphasis(num_feats=n_feats, alpha=pre_emphasis_alpha) if use_pre_emphasis else None
        self.pre_conv=Sequential(
            nn.Conv1d(n_feats, input_channels,kernel_size=5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(input_channels)
        )
        self.se_res_2_list=nn.ModuleList([
            SeRes2Block(input_channels, reduction_dim, num_groups=4,k=3, d=2),
            SeRes2Block(input_channels, reduction_dim, num_groups=4,k=3, d=3),
            SeRes2Block(input_channels, reduction_dim, num_groups=4,k=3, d=4),
        ]
        )
        self.output_net=nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1),#(B, hidden_channels, d)
            AttnStatPool(hidden_channels, reduction_dim),#(B, hidden_channels*2)
            nn.BatchNorm1d(hidden_channels*2),
            nn.Linear(hidden_channels*2,output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, spectral_feat, **batch):
        """
        Forward pass for custom implementation.

        Args:
            spectral_feat (Tensor): input spectral features (B, n_feats, T).
        Returns:
            output (dict): output dict containing logits and embedding (before last linear).
        """
        x = spectral_feat
        if self.use_pre_emphasis:
            x = self.pre_emphasis(x)
        x = self.pre_conv(x)
        outputs=[]
        for layer in self.se_res_2_list:
            x=layer(x)
            outputs.append(x)
        combined_output=torch.cat(outputs, dim=-1)
        embedding=self.output_net[:-1](combined_output)
        logits=self.output_net[-1](embedding)
        return {"logits": logits, "embedding": embedding}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class EcappaTDNNPretrained(nn.Module):
    """
    Pretrained ECAPPA-TDNN from SpeechBrain.
    Forward takes raw audio as input.
    """
    def __init__(self):
        """Load pretrained ECAPPA-TDNN from SpeechBrain."""
        super().__init__()
        print("Loading pretrained ECAPPA-TDNN model from SpeechBrain...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Successfully loaded pretrained ECAPPA-TDNN from SpeechBrain")

    def forward(self, audio, **batch):
        """
        Forward pass for pretrained model.

        Args:
            audio (Tensor): raw audio signal (B, audio_length) at 16kHz sampling rate.
        Returns:
            output (dict): output dict containing embeddings.
        """
        assert audio.shape[0]>=2
        embeddings = self.classifier.encode_batch(audio).squeeze(1)        
        return {"embedding": embeddings, "logits": embeddings}


