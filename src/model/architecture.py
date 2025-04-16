import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.mps
from monai.networks.nets.swin_unetr import SwinTransformer

class SwinUNETRFallback(nn.Module): #use monai swintransformer encoder, return last feature map, upsample to 128^3, then 1x1 conv
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        spatial_dims=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(4, 4, 4),
            patch_size=(4, 4, 4),
            depths=(2, 4, 6, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        self.out_conv = nn.Conv3d(feature_size * 16, out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        #up_feats = F.interpolate(feats, scale_factor=64, mode="trilinear", align_corners=True) too heavy for system
        up_feats = F.interpolate(feats, scale_factor=8, mode="trilinear", align_corners=True)
        up_feats = F.interpolate(up_feats, scale_factor=8, mode="trilinear", align_corners=True) #upsample in 2 steps for memory
        logits = self.out_conv(up_feats)
        return logits
