import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CUNET(nn.Module):
    """
    Input: [B, C, F, T]
    Output: [B, C, T, F]
    """
    def __init__(self):
        super(CUNET, self).__init__()
        self.conv_block_1 = CausalConvBlock(8, 32, (6, 2), (2, 1))
        self.conv_block_2 = CausalConvBlock(32, 32, (7, 2), (2, 1))
        self.conv_block_3 = CausalConvBlock(32, 64, (7, 2), (2, 1))
        self.conv_block_4 = CausalConvBlock(64, 64, (6, 2), (2, 1))
        self.conv_block_5 = CausalConvBlock(64, 96, (6, 2), (2, 1))
        self.conv_block_6 = CausalConvBlock(96, 96, (6, 2), (2, 1))
        self.conv_block_7 = CausalConvBlock(96, 128, (2, 2), (2, 1))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        self.tran_conv_block_1 = CausalTransConvBlock(256, 256, (2, 2), (1, 1))
        self.tran_conv_block_2 = CausalTransConvBlock(256 + 128, 128, (2, 2), (2, 1))
        self.tran_conv_block_3 = CausalTransConvBlock(128 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_4 = CausalTransConvBlock(96 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_5 = CausalTransConvBlock(96 + 64, 64, (6, 2), (2, 1))
        self.tran_conv_block_6 = CausalTransConvBlock(64 + 64, 64, (7, 2), (2, 1))
        self.tran_conv_block_7 = CausalTransConvBlock(64 + 32, 32,  (7, 2), (2, 1))
        self.tran_conv_block_8 = CausalTransConvBlock(32 + 32, 32,  (6, 2), (2, 1))
        self.last_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=8,
                kernel_size=1,
                stride=1),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)) # author of paper said do this
        self.dense = nn.Sequential(nn.Linear(514, 514),nn.Sigmoid())

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
        d = self.last_conv_block(d)
        d = d.permute(0,1,3,2)[:,:,:-8] # author of paper said do this
        d = self.dense(d).permute(0,1,3,2)
        return d

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)


def snr_loss(inputs, label):
    return -(si_snr(inputs, label))

def MAE_loss(inputs, label):
    diff = torch.abs(inputs - label)
    return diff.sum()/inputs.shape[-1]


if __name__ == '__main__':
    layer = CUNET()
    K = 8 # zeros padding frame number
    x = torch.rand(1, 8, 514, 249) #the 4 seconds original input
    print("input shape:", x.shape)
    prefix_frames = torch.zeros(1, 8, 514, K) # K zeros prefix frames 
    y=torch.cat(( x,prefix_frames),3)
    print("output shape:",layer(y).shape)
    total_num = sum(p.numel() for p in layer.parameters())
    print(total_num)
