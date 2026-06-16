from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, cfg

logger = logging.getLogger(__name__)



class PerturbationUNet(nn.Module):

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Two conv layers with GroupNorm and ReLU activations"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def _down(self, in_ch: int, out_ch: int) -> tuple[nn.Sequential, nn.MaxPool2d]:
        """Encoder: conv block + 2x max-pool"""
        return self._conv_block(in_ch, out_ch), nn.MaxPool2d(2)

    def _up(self, in_ch: int, out_ch: int) -> tuple[nn.Upsample, nn.Sequential]:
        """Decoder: 2x bilinear upsample + conv block"""
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), self._conv_block(in_ch, out_ch)

    def __init__(self, base_channels: int = 8, epsilon: float = 8 / 255.0):
        super().__init__()

        self.epsilon = epsilon
        c = base_channels 

        # Encoder
        self.enc1_conv, self.pool1 = self._down(3, c)           # (3) -> (c)
        self.enc2_conv, self.pool2 = self._down(c, c * 2)       # (c) -> (2c)
        self.enc3_conv, self.pool3 = self._down(c * 2, c * 4)   # (2c) -> (4c)

        # Bottleneck
        self.bottleneck = self._conv_block(c * 4, c * 8)  # (4c) -> (8c)

        # Decoder
        self.up3, self.dec3 = self._up(c * 8, c * 4) # (8c) -> (4c)
        self.up2, self.dec2 = self._up(c * 4, c * 2) # (4c) -> (2c)
        self.up1, self.dec1 = self._up(c * 2, c) # (2c) -> (c)

        # Output projection to RGB perturbation
        self.head = nn.Conv2d(c, 3, kernel_size=1)  # (c) -> (3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        s1 = self.enc1_conv(x) # (B, c, H, W)
        s2 = self.enc2_conv(self.pool1(s1)) # (B, 2c, H/2, W/2)
        s3 = self.enc3_conv(self.pool2(s2)) # (B, 4c, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool3(s3)) # (B, 8c, H/8, W/8)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1)) # (B, 4c, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1)) # (B, 2c, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1)) # (B, c, H, W)

        # Bound, clip, and add the perturbation to the original image
        raw = self.head(d1)
        perturbation = self.epsilon * torch.tanh(raw)
        adv_image = torch.clamp(x + perturbation, 0.0, 1.0)

        return adv_image, perturbation

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_perturbation_net(config: Config = cfg) -> PerturbationUNet:
    """
    Builds and returns the perturbation UNet based on the provided configuration.
    
    Args:
        config: An instance of the Config dataclass containing model hyperparameters.
    
    Returns:
        An instance of the PerturbationUNet model.
    """
    net = PerturbationUNet(
        base_channels=config.unet_base_channels,
        epsilon=config.epsilon,
    )
    logger.info(f"PerturbationUNet | parameters: {net.num_parameters:,}")
    return net