import torch
import numpy as np
import torch.nn as nn

import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class RecoveryNet(nn.Module):
    def __init__(self, config):
        super(RecoveryNet, self).__init__()
        self.config = config
        self.pool = nn.MaxPool2d(2)

        self.conv_1_64 = ConvBlock(1, 32)
        self.conv_64_128 = ConvBlock(32, 64)

        self.conv_128_256_128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )
        self.conv_64_32 = ConvBlock(64, 32)

        self.conv_128_64 = ConvBlock(64, 32)
        self.conv_64_1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x_64 = self.conv_1_64(x)
        x_128 = self.conv_64_128(self.pool(x_64))
        y = self.conv_128_256_128(x_128)
        y = self.conv_64_32(self.up_sample(y))
        y = torch.cat((y, x_64), dim=-3)
        y = self.conv_128_64(y)
        y = self.conv_64_1(y)
        return y


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.matrix_size = 32
        self.idx = self.config.block_size // self.matrix_size
        points = self.matrix_size ** 2
        phi_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(int(config.ratio * points), points))
        q_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(points, int(config.ratio * points)))

        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(q_init).float(), requires_grad=True)

        self.denoising = models.IterLayers(dim=8 ** 2, depth=6, dropout=0.)
        self.recovery = RecoveryNet(config)

        self.bridge = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        eta_init = 0.1 * np.ones(shape=(self.config.block_size, self.config.block_size))
        self.eta = nn.Parameter(torch.from_numpy(eta_init).float(), requires_grad=True)

    def forward(self, inputs):
        y = self.sampling(inputs, self.matrix_size)
        recon = self.recon(y, self.matrix_size, inputs.size(0))
        return recon

    def sampling(self, inputs, init_block):
        inputs = inputs.to(self.config.device)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=2), dim=0)

        inputs = torch.reshape(inputs, [-1, init_block ** 2])
        inputs = torch.transpose(inputs, 0, 1)
        y = torch.matmul(self.phi, inputs)
        return y

    def init(self, y, matrix, init_block, batch_size):
        x_init = torch.matmul(matrix, y)
        x_init_32 = torch.transpose(x_init, 0, 1).reshape([-1, 1, init_block, init_block])
        x_init = torch.cat(torch.split(x_init_32, split_size_or_sections=self.idx * batch_size, dim=0), dim=2)
        x_init_block = torch.cat(torch.split(x_init, split_size_or_sections=batch_size, dim=0), dim=3)
        return x_init_32, x_init_block

    def recon(self, y, init_block, batch_size):
        x_init_32, x_init_block = self.init(y, self.Q, init_block, batch_size)

        x_tmp = self.block2patch(x_init_block, 8)
        x_tmp, attn = x_tmp - self.denoising(x_tmp)[0], self.denoising(x_tmp)[1]
        x_tmp = self.patch2block(x_tmp, self.config.block_size, 8, 1)

        recon = x_init_block - self.recovery(x_init_block)

        x_out = recon - torch.matmul(self.eta, self.bridge(torch.cat((x_tmp, recon), dim=1)))
        return x_out, _

    def block2patch(self, inputs, size):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=size, dim=3), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=size, dim=2), dim=1)
        return inputs

    def patch2block(self, inputs, ori_size, size, channel):
        assert ori_size % size == 0, f"original {ori_size} should be divided by size {size}."
        idx = int(ori_size / size)
        outputs = torch.cat(torch.split(inputs, split_size_or_sections=channel * idx, dim=1), dim=2)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=channel, dim=1), dim=3)
        return outputs
