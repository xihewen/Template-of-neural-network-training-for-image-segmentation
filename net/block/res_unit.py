import torch.nn as nn
import torch


# 特征图计算公式 y=ceil((x+2p-k-(k-1)*(d-1)+1)/s)    N = (W − k + 2P +1)/S

def BN_ReLU_Conv3D(ch_in, ch_out, kernel_size=3, stride=1, padding=1,
                   bias=False):  # k = 3,s = 1,p = 1特征图不变 k = 3,s = 2,p = 1减半
    """Performs a batch normalization followed by a ReLU6."""
    BN_ReLU_Conv = nn.Sequential(
        nn.BatchNorm3d(ch_in),
        nn.ReLU6(inplace=True),
        nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )
    return BN_ReLU_Conv



def BN_ReLU_Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1,
                   bias=False):  # k = 3,s = 1,p = 1特征图不变 k = 3,s = 2,p = 1减半
    BN_ReLU_Conv = nn.Sequential(
        nn.BatchNorm2d(ch_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )
    """Performs a batch normalization followed by a ReLU6."""
    return BN_ReLU_Conv


def Conv3D(ch_in, ch_out, kernel_size=1, stride=2, padding=0, bias=False):  # k = 1,s = 2,p = 0 ## k = 3,s = 2,p = 1 特征图减半
    """Performs 3D convolution without bias and activation function."""

    return nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def Conv2D(ch_in, ch_out, kernel_size=1, stride=2, padding=0, bias=False):  # k = 1,s = 1,p = 0 ## k = 3,s = 1,p = 1 特征图不变
    """Performs 3D convolution without bias and activation function."""

    return nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


# N=(w-1)×s+k-2p
def TConv3D(ch_in, ch_out, kernel_size=3, stride=2, padding=0, bias=False):  # k = 2,s = 2,p = 0特征图2倍  2, 1, 0不变
    """Performs 3D convolution without bias and activation function."""

    return nn.ConvTranspose3d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)


if __name__ == '__main__':
    tconv = TConv3D(1, 1, 2, 2, 0)
    inputs = torch.rand(1, 2, 6, 6).unsqueeze(0)
    print(inputs.size())
    print(tconv(inputs).size())
