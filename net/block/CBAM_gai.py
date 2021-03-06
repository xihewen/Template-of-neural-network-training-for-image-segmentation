###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
import torch
from torch.nn import Module, Conv3d, Parameter, Softmax

torch_ver = torch.__version__[:3]

# 通道注意力机制
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, ch_in=32, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(ch_in, ch_in // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(ch_in // ratio, ch_in, 1, bias=False)

        self.gamma1 = Parameter(torch.zeros(1))

        self.gamma2 = Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.gamma1 * avg_out + self.gamma2 * max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class conbination_attention_block_wei(Module):
    def __init__(self, ch_in=32, kernel_size=3):
        super().__init__()
        # 调用方法：其他模块中调用，注意oup是模块输入通道数，严格和你输入到 ChannelAttention模块中的前一个模块或者操作的输出通道数一致，SpatialAttention 没有限制
        self.ca = ChannelAttention(ch_in)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : position attention  + channel attention
        """
        # 具体实现 ，x表示数据，来自当前层或者某一层数据
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


if __name__ == '__main__':
    inputs = torch.rand(2, 32, 256, 256)
    print(inputs)
    outputs = torch.max(inputs, -1, keepdim=True)[0].expand_as(inputs) - inputs
    print(torch.max(inputs, -1, keepdim=True)[0])
    print(torch.max(inputs, -1, keepdim=True)[0].size(),
          torch.max(inputs, -1, keepdim=True)[0].expand_as(inputs).size())
    print(torch.max(inputs, -1, keepdim=True)[0].expand_as(inputs))
    print(torch.max(inputs, -1, keepdim=True)[0].expand_as(inputs) - inputs)
    print(inputs.size())

    net = PAM_Module(32)
    res = net(inputs)
    # [batch, h, w, num_heads, channels / num_heads]
    print('res shape:', res.shape)
