import torch
from torch import nn


class Bottleneck(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, stride):
        super().__init__()
            
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)

        self.conv2 = nn.Conv2d(midplanes, midplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(midplanes, outplanes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        nn.init.zeros_(self.bn3.weight)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(*[nn.AvgPool2d(stride),
                                              nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                              nn.BatchNorm2d(outplanes)])


    def forward(self, x: torch.Tensor):
        
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.avgpool(out)

        out = self.bn3(self.conv3(out))

        identity = self.downsample(x) if self.downsample else x
                        
        out += identity

        out = self.relu(out)

        return out


