import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, ResidualBlock, n_class):
        super(ResNet50, self).__init__()
        in_channels = [3, 64, 128, 256, 512, 1024]
        out_channels = [64, 128, 256, 512, 1024, n_class]
        self.__conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.__layers = []
        for i in range(1, 5):
            self.__layers.append(self.__make_layer(ResidualBlock, in_channels[i], out_channels[i], 3))
        self.__layers = nn.Sequential(*self.__layers)
        self.__fc = nn.Linear()

    def __make_layer(self, block, in_channel, out_channel, n_block, stride=1):
        layers = []
        strides = [stride] + [1] * (n_block - 1)
        for stride in strides:
            layers.append(block(in_channel, out_channel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.__conv1(x)
        out = self.__layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.__fc(out)
        return out
