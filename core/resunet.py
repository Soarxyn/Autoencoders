from typing import List
import core.modules as m
import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DecoderBlock, self).__init__()

        self.upsample = m.TransposedConv2d(input_dim, input_dim, 2, 2)
        self.conv = m.ResidualConv2d(input_dim + output_dim, output_dim, 1, 1)

    def forward(self, x, skip):
        output = self.upsample(x)
        output = torch.cat([output, skip], dim=1)
        output = self.conv(output)

        return output


class ResUNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 2,
        filters: List[int] = [32, 64, 128, 256],
        logits: bool = False,
    ) -> None:
        super(ResUNet, self).__init__()

        self.input_layer = m.ResidualConv2d(input_dim, filters[0], 1, 1, True)

        self.residual_block_1 = m.ResidualConv2d(filters[0], filters[1], 2, 1)
        self.residual_block_2 = m.ResidualConv2d(filters[1], filters[2], 2, 1)

        self.bridge = m.ResidualConv2d(filters[2], filters[3], 2, 1)

        self.decoder_1 = DecoderBlock(filters[3], filters[2])
        self.decoder_2 = DecoderBlock(filters[2], filters[1])
        self.decoder_3 = DecoderBlock(filters[1], filters[0])

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], output_dim, 1, 1),
            nn.Sigmoid() if not logits else nn.Identity(),
        )

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.residual_block_1(x0)
        x2 = self.residual_block_2(x1)
        x3 = self.bridge(x2)
        d1 = self.decoder_1(x3, x2)
        d2 = self.decoder_2(d1, x1)
        d3 = self.decoder_3(d2, x0)
        output = self.output_layer(d3)

        return output
