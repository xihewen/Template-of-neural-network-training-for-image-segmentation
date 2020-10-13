"""

基础网络脚本
"""
import torch.nn as nn


class Two_Attention_block(nn.Module):  # 通道数 out    F_g=down_in_channels, F_l=down_in_channels, F_int=in_channels
    def __init__(self, ch_in_g, ch_in_x, F_int=4):  # F_g = F_l, F_int = 下一级
        super(Two_Attention_block, self).__init__()
        self.W_g = nn.Sequential(  # out = F_int
            nn.Conv3d(ch_in_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(  # out = F_int
            nn.Conv3d(ch_in_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi_int = nn.Sequential(  # out = 1
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)  # 输出通道数 out
        # 上采样的 l 卷积
        x1 = self.W_x(x)  # 输出通道数 out
        # concat + relu
        psi = self.relu(g1 + x1)  # out =  F_int   数值相加 通道数不变  out
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi_int(psi)  # out =  1个向量
        # 返回加权的 x
        return x * psi  # 通道数 out
