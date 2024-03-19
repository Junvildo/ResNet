from torch import nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

seq = nn.Sequential


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

def conv2d(ch_in, ch_out, kz, s=1, p=0):
    return seq(
        spectral_norm(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kz, stride=s, padding=p)),
        spectral_norm(nn.BatchNorm2d(ch_out)),
    )

def conv1():
    return seq(
        conv2d(ch_in=3, ch_out=64, kz=7, s=2, p=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )

class ResNet_18(nn.Module):
    def __init__(self, num_class=1000) -> None:
        super().__init__()
        self.skip64 = conv2d(ch_in=64, ch_out=128, kz=1, s=2, p=0)
        self.skip128 = conv2d(ch_in=128, ch_out=256, kz=1, s=2, p=0)
        self.skip256 = conv2d(ch_in=256, ch_out=512, kz=1, s=2, p=0)
        self.con2_x = seq(
            conv2d(ch_in=64, ch_out=64, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=64, ch_out=64, kz=3, s=1, p=1),
        )
        self.con3_x_a = seq(
            conv2d(ch_in=64, ch_out=128, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=128, ch_out=128, kz=3, s=1, p=1),
        )
        self.con3_x_b = seq(
            conv2d(ch_in=128, ch_out=128, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=128, ch_out=128, kz=3, s=1, p=1),
        )
        self.con4_x_a = seq(
            conv2d(ch_in=128, ch_out=256, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=256, ch_out=256, kz=3, s=1, p=1),
        )
        self.con4_x_b = seq(
            conv2d(ch_in=256, ch_out=256, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=256, ch_out=256, kz=3, s=1, p=1),
        )
        self.con5_x_a = seq(
            conv2d(ch_in=256, ch_out=512, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1),
        )
        self.con5_x_b = seq(
            conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1),
        )
        self.InitBlock = seq(
            conv1(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = seq(
            nn.AvgPool2d(kernel_size=7, ceil_mode=False, stride=7, padding=0, count_include_pad=True),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        init_val = self.InitBlock(x)
        conv2_x = self.con2_x(init_val) + init_val
        nn.ReLU()(conv2_x)
        conv2_x = self.con2_x(conv2_x) + conv2_x
        nn.ReLU()(conv2_x)
        conv3_x = self.con3_x_a(conv2_x) + self.skip64(conv2_x)
        nn.ReLU()(conv3_x)
        conv3_x = self.con3_x_b(conv3_x) + conv3_x
        nn.ReLU()(conv3_x)
        conv4_x = self.con4_x_a(conv3_x) + self.skip128(conv3_x)
        nn.ReLU()(conv4_x)
        conv4_x = self.con4_x_b(conv4_x) + conv4_x
        nn.ReLU()(conv4_x)
        conv5_x = self.con5_x_a(conv4_x) + self.skip256(conv4_x)
        nn.ReLU()(conv5_x)
        conv5_x = self.con5_x_b(conv5_x) + conv5_x
        nn.ReLU()(conv5_x)
        res = self.fc(conv5_x)
        return res
    
class ResNet_50(nn.Module):
    def __init__(self, num_class=1000) -> None:
        super().__init__()
        self.InitBlock = seq(
            conv1(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = seq(
            nn.AvgPool2d(kernel_size=7, ceil_mode=False, stride=7, padding=0, count_include_pad=True),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_class),
            nn.Sigmoid()
        )
        self.conv2_x_a = seq(
            conv2d(ch_in=64, ch_out=64, kz=1, s=1, p=0), nn.ReLU(),
            conv2d(ch_in=64, ch_out=64, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=64, ch_out=256,kz=1,s=1,p=0),
        )
        self.conv2_x_b = seq(
            conv2d(ch_in=256, ch_out=64, kz=1, s=1, p=0), nn.ReLU(),
            conv2d(ch_in=64, ch_out=64, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=64, ch_out=256,kz=1,s=1,p=0),
        )
        self.skip112_56 = conv2d(ch_in=64, ch_out=256, kz=1, s=1, p=0)
        self.conv3_x_a = seq(
            conv2d(ch_in=256, ch_out=128,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=128, ch_out=128, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=128, ch_out=512, kz=1, s=1, p=0),
        )
        self.conv3_x_b = seq(
            conv2d(ch_in=512, ch_out=128,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=128, ch_out=128, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=128, ch_out=512, kz=1, s=1, p=0),
        )
        self.skip56_28 = conv2d(ch_in=256, ch_out=512, kz=1, s=2, p=0)
        self.conv4_x_a = seq(
            conv2d(ch_in=512, ch_out=256,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=256, ch_out=256, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=256, ch_out=1024, kz=1, s=1, p=0),
        )
        self.conv4_x_b = seq(
            conv2d(ch_in=1024, ch_out=256,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=256, ch_out=256, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=256, ch_out=1024, kz=1, s=1, p=0),
        )
        self.skip28_14 = conv2d(ch_in=512, ch_out=1024, kz=1, s=2, p=0)
        self.conv5_x_a = seq(
            conv2d(ch_in=1024, ch_out=512,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=512, ch_out=512, kz=3, s=2, p=1), nn.ReLU(),
            conv2d(ch_in=512, ch_out=2048, kz=1, s=1, p=0),
        )
        self.conv5_x_b = seq(
            conv2d(ch_in=2048, ch_out=512,kz=1,s=1,p=0), nn.ReLU(),
            conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
            conv2d(ch_in=512, ch_out=2048, kz=1, s=1, p=0),
        )
        self.skip14_7 = conv2d(ch_in=1024, ch_out=2048, kz=1, s=2, p=0)

    def forward(self, x):
        # Init
        init_val = self.InitBlock(x)

        # Con2_x
        conv2_x = self.conv2_x_a(init_val) + self.skip112_56(init_val)
        nn.ReLU()(conv2_x)
        conv2_x = self.conv2_x_b(conv2_x) + conv2_x
        nn.ReLU()(conv2_x)
        conv2_x = self.conv2_x_b(conv2_x) + conv2_x
        nn.ReLU()(conv2_x)

        # Conv3_x
        conv3_x = self.conv3_x_a(conv2_x) + self.skip56_28(conv2_x)
        nn.ReLU()(conv3_x)
        conv3_x = self.conv3_x_b(conv3_x) + conv3_x
        nn.ReLU()(conv2_x)
        conv3_x = self.conv3_x_b(conv3_x) + conv3_x
        nn.ReLU()(conv3_x)
        conv3_x = self.conv3_x_b(conv3_x) + conv3_x
        nn.ReLU()(conv3_x)

        # Conv4_x
        conv4_x = self.conv4_x_a(conv3_x) + self.skip28_14(conv3_x)
        nn.ReLU()(conv4_x)
        conv4_x = self.conv4_x_b(conv4_x) + conv4_x
        nn.ReLU()(conv4_x)
        conv4_x = self.conv4_x_b(conv4_x) + conv4_x
        nn.ReLU()(conv4_x)
        conv4_x = self.conv4_x_b(conv4_x) + conv4_x
        nn.ReLU()(conv4_x)


        # Conv5_x
        conv5_x = self.conv5_x_a(conv4_x) + self.skip14_7(conv4_x)
        nn.ReLU()(conv5_x)
        conv5_x = self.conv5_x_b(conv5_x) + conv5_x
        nn.ReLU()(conv5_x)
        conv5_x = self.conv5_x_b(conv5_x) + conv5_x
        nn.ReLU()(conv5_x)
        conv5_x = self.conv5_x_b(conv5_x) + conv5_x
        nn.ReLU()(conv5_x)

        # FC
        result = self.fc(conv5_x)

        return result