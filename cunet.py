""" Yangang Cao 2021.12.1 15:30"""

import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1].contiguous()  # chomp size
        x = self.norm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1].contiguous()  # chomp size
        x = self.norm(x)
        x =self.dropout(x)
        x = self.activation(x)
        return x


class CUNET(nn.Module):
    """
    Input: [B, C, F, T]
    Output: [B, C, T, F]
    """
    def __init__(self):
        super(CUNET, self).__init__()
        self.conv_block_1 = CausalConvBlock(8, 32, (6, 2), (2, 1), (2, 1))
        self.conv_block_2 = CausalConvBlock(32, 32, (6, 2), (2, 1), (2, 1))
        # kernel_size of conv_block_3 is (7,2) in paper, just for avoiding padding in Tensorflow implementation, 
        # I think (6,2) in PyTorch is OK, but padding is ugly
        self.conv_block_3 = CausalConvBlock(32, 64, (6, 2), (2, 1), (2, 1))
        self.conv_block_4 = CausalConvBlock(64, 64, (6, 2), (2, 1), (2, 1))
        self.conv_block_5 = CausalConvBlock(64, 96, (6, 2), (2, 1), (2, 1))
        self.conv_block_6 = CausalConvBlock(96, 96, (6, 2), (2, 1), (2, 1))
        self.conv_block_7 = CausalConvBlock(96, 128, (2, 2), (2, 1), (0, 1))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1), (0, 1))

        self.tran_conv_block_1 = CausalTransConvBlock(256, 256, (2, 2), (1, 1), (0, 0))
        self.tran_conv_block_2 = CausalTransConvBlock(256 + 128, 128, (2, 2), (2, 1), (0,0))
        self.tran_conv_block_3 = CausalTransConvBlock(128 + 96, 96, (6, 2), (2, 1), (2, 0))
        self.tran_conv_block_4 = CausalTransConvBlock(96 + 96, 96, (6, 2), (2, 1), (2, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(96 + 64, 64, (6, 2), (2, 1), (2, 0))
        self.tran_conv_block_6 = CausalTransConvBlock(64 + 64, 64, (6, 2), (2, 1), (2, 0))
        self.tran_conv_block_7 = CausalTransConvBlock(64 + 32, 32,  (6, 2), (2, 1), (2, 0))
        # I change the output channel into 8, and 32 in paper
        self.tran_conv_block_8 = CausalTransConvBlock(32 + 32, 8,  (6, 2), (2, 1), (2, 0))
        self.dense = nn.Linear(512, 512)


    def forward(self, x):
        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        d = self.tran_conv_block_1(e8)
        d = self.tran_conv_block_2(torch.cat((d, e7), 1))
        d = self.tran_conv_block_3(torch.cat((d, e6), 1))
        d = self.tran_conv_block_4(torch.cat((d, e5), 1))
        d = self.tran_conv_block_5(torch.cat((d, e4), 1))
        d = self.tran_conv_block_6(torch.cat((d, e3), 1))
        d = self.tran_conv_block_7(torch.cat((d, e2), 1))
        d = self.tran_conv_block_8(torch.cat((d, e1), 1))
        d = d.permute(0,1,3,2)
        d = self.dense(d)
        return d


if __name__ == '__main__':
    layer = CUNET()
    x = torch.rand(1, 8, 512, 249)
    print("input shape:", x.shape)
    print("output shape:",layer(x).shape)
    total_num = sum(p.numel() for p in layer.parameters())
    print(total_num)
