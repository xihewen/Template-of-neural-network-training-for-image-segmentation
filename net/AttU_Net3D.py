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


class pub(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(pub, self).__init__()

        self.pub = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.pub(x)


class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(unet3dDown, self).__init__()
        self.pub = pub(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        xh = self.pub(x)

        xs = self.pool(xh)

        return xh, xs


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, sample=True):
        super(unet3dUp, self).__init__()
        self.half_channel = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.pub = pub(out_channels, out_channels)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='trilinear')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.attention = Two_Attention_block(in_channels // 2,in_channels // 2)
    def forward(self, x, x1):
        x = self.half_channel(x)
        x = self.sample(x)
        x = self.attention(x)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, init_channels=1, class_nums=1, sample=True):
        super(unet3d, self).__init__()
        self.down1 = unet3dDown(init_channels, 32)
        self.down2 = unet3dDown(32, 64)
        self.down3 = unet3dDown(64, 128)
        self.down4 = unet3dDown(128, 256)
        self.bottle = pub(256, 512)

        self.up4 = unet3dUp(512, 256, sample)
        self.up3 = unet3dUp(256, 128, sample)
        self.up2 = unet3dUp(128, 64, sample)
        self.up1 = unet3dUp(64, 32, sample)
        self.con_last = nn.Conv3d(32, class_nums, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xh1, xs1 = self.down1(x)
        xh2, xs2 = self.down2(xs1)
        xh3, xs3 = self.down3(xs2)
        xh4, xs4 = self.down4(xs3)
        x = self.bottle(xs4)
        x = self.up4(x, xh4)
        x = self.up3(x, xh3)
        x = self.up2(x, xh2)
        x = self.up1(x, xh1)
        x = self.con_last(x)
        return self.sigmoid(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


net = unet3d(init_channels=1, class_nums=1, sample=True)

# 计算网络参数
print('net total parameters:', sum(param.numel() for param in net.parameters()))
