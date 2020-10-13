"""

训练脚本
"""

import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net
import torch.nn as nn
import parameter as para
from logger.logger import Logger

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# 定义网络
net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义Dateset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

# 定义数据加载
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# 挑选损失函数
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[0]

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# 深度监督衰减系数
alpha = para.alpha

# 训练网络
start = time()
upsample = nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear')
logger = Logger('./log/')
for epoch in range(para.Epoch):

    lr_decay.step()

    mean_loss = []
    mean_loss1 = []
    mean_loss2 = []
    mean_loss3 = []

    loss = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(upsample(outputs[0]), seg)
        loss2 = loss_func(upsample(outputs[1]), seg)
        loss3 = loss_func(upsample(outputs[2]), seg)

        loss = (loss1 + loss2) * alpha + loss3

        mean_loss.append(loss.item())
        mean_loss1.append(loss1.item())
        mean_loss2.append(loss2.item())
        mean_loss3.append(loss3.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(
        'epoch:{},loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f},  time:{:.3f} min'.format(epoch, loss1.item(), loss2.item(),
                                                                                     loss3.item(),
                                                                                     (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)
    mean_loss1 = sum(mean_loss1) / len(mean_loss1)
    mean_loss2 = sum(mean_loss2) / len(mean_loss2)
    mean_loss3 = sum(mean_loss3) / len(mean_loss3)

    # 保存模型
    if epoch % 50 is 0 and epoch is not 0:
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss3, mean_loss3))

    # 对深度监督系数进行衰减
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8

    logger.scalar_summary('mean_loss3', mean_loss3, epoch)
    logger.scalar_summary('mean_loss1', mean_loss1, epoch)
    logger.scalar_summary('mean_loss2', mean_loss2, epoch)
    logger.scalar_summary('mean_loss', mean_loss, epoch)
# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
