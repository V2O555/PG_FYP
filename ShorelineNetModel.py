import torch
from torch import nn
import torchvision.models as models


class unet_model(nn.Module):
    def __init__(self, n_classes):
        super(unet_model, self).__init__()

        # Encoder
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True).features
        self.layer_names = [
            '0',  # block_1_expand_relu: 64 x 64 x 96
            '3',  # block_3_expand_relu: 32 x 32 x 144
            '6',  # block_6_expand_relu: 16 x 16 x 192
            '13',  # block_13_expand_relu: 8  x 8  x 576
            '16',  # block_16_project:    4  x 4  x 320
        ]
        self.layers = [int(name) for name in self.layer_names]

        # Decoder
        self.up_stack = nn.ModuleList([
            self.deconv_block(160, 512, dropout_rate=0.5),  # 4x4 -> 8x8
            self.deconv_block(608, 256, dropout_rate=0.4),  # 8x8 -> 16x16
            self.deconv_block(288, 128, dropout_rate=0.3),  # 16x16 -> 32x32
            self.deconv_block(152, 64, dropout_rate=0.2),  # 32x32 -> 64x64
        ])

        # Final layer
        self.final = nn.ConvTranspose2d(96, n_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def deconv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                     dropout_rate=0.5):
        '''
        ConvTranspose2d => BatchNorm2d => Dropout => ReLU
        '''
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False, output_padding=output_padding))
        layers.append(nn.BatchNorm2d(out_channels))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        skips = []
        for i, layer in enumerate(self.mobilenet_v2):
            x = layer(x)
            if i in self.layers:
                skips.append(x)

        # Decoder
        x = skips[-1]
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        # Final layer
        x = self.final(x)
        return x
