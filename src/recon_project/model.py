from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, base_channels: int = 32, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        c = base_channels

        self.enc1 = ConvBlock(1, c)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.down2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 2, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        delta = self.out(d1)

        y = x + delta if self.residual else delta
        return torch.clamp(y, 0.0, 1.0)


class ResidualCNN(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.net(x), 0.0, 1.0)


def build_model(name: str) -> nn.Module:
    if name == "unet_small":
        return UNetSmall(base_channels=32, residual=True)
    if name == "unet_tiny":
        return UNetSmall(base_channels=16, residual=True)
    if name == "residual_cnn":
        return ResidualCNN(channels=64, depth=6)
    raise ValueError("Unknown model {!r}; choose unet_tiny, unet_small, or residual_cnn".format(name))

