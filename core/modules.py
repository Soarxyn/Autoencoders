import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int = 1,
        padding: int = 1,
        stem_block: bool = False,
        activation: type = nn.ReLU,
    ) -> None:
        super(ResidualConv2d, self).__init__()

        self.input_block = nn.Identity()

        if not stem_block:
            self.input_block = nn.Sequential(nn.BatchNorm2d(input_dim), activation())

        self.convolution_block = nn.Sequential(
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
            activation(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

        self.convolution_skip = nn.Sequential(
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.convolution_block(self.input_block(x)) + self.convolution_skip(x)


class TransposedConv2d(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, kernel_size: int = 1, stride: int = 1
    ) -> None:
        super(TransposedConv2d, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
