# Adapted from
# https://github.com/facebookresearch/sapiens2/blob/main/sapiens/dense/src/models/heads/normal_head.py
# (Copyright (c) Meta Platforms, Inc. and affiliates.), stripped of the
# sapiens.registry/MODELS decorator and the loss-building/`.loss()` code path,
# since we only need inference (`forward()`). Architecture and weight names
# are otherwise unchanged so pretrained checkpoints load directly.

from typing import List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.init import trunc_normal_


class NormalHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        channels: int = 16,
        upsample_channels: List[int] = [768, 384, 192, 96],
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self._build_network(upsample_channels, conv_out_channels, conv_kernel_sizes)
        in_channels = (
            conv_out_channels[-1] if conv_out_channels else upsample_channels[-1]
        )
        self.conv_normal = nn.Conv2d(in_channels, 3, kernel_size=1)
        self._init_weights()

    def _build_network(
        self,
        upsample_channels: List[int],
        conv_out_channels: Optional[Sequence[int]],
        conv_kernel_sizes: Optional[Sequence[int]],
    ) -> None:
        in_channels = self.in_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        up_blocks = []
        cur_ch = in_channels
        for out_ch in upsample_channels:
            up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(cur_ch, out_ch * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.InstanceNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )
            cur_ch = out_ch
        self.upsample_blocks = nn.Sequential(*up_blocks)

        conv_layers = []
        if conv_out_channels and conv_kernel_sizes:
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                conv_layers.extend(
                    [
                        nn.Conv2d(cur_ch, out_ch, k, padding=(k - 1) // 2),
                        nn.InstanceNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    ]
                )
                cur_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_out", nonlinearity="relu"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_in", nonlinearity="linear"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        x_normal = self.input_conv(x)
        x_normal = self.upsample_blocks(x_normal)
        x_normal = self.conv_layers(x_normal)
        normal = self.conv_normal(x_normal)
        return normal
