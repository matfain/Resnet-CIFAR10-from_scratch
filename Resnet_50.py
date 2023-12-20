import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torchsummary import summary

# floor((W - F + 2P)/S + 1)         

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        self.conv1= nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv2= nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3= nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.skip= nn.Sequential()

        if downsample or in_channels != out_channels:
            self.skip= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Initialization
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        if len(self.skip) > 0:
            init.kaiming_normal_(self.skip[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        skip= self.skip(x)
        var1= F.relu(self.bn1(self.conv1(x)))
        var2= F.relu(self.bn2(self.conv2(var1)))
        var3= F.relu(self.bn3(self.conv3(var2)))
        out= var3 + skip
        return F.relu(out)
    
class Layer0(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size= 5, stride=1, padding= 2)     # (32-5+4)/1 + 1 = 32 -1 + 1 = 32
        self.bn1 = nn.BatchNorm2d(64)

        # Initialization
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Final_layer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        gap = self.gap(x)
        x_flat = gap.view(gap.size(0), -1)
        logits = self.fc(x_flat)
        return logits
    

class ResNet50(nn.Module):
    def __init__(self, in_channels, layer0, bottleneckblock, final_layer, num_classes):
        super().__init__()

        self.layer0= nn.Sequential(layer0(in_channels))

        self.layer1= nn.Sequential(
            bottleneckblock(64, 256, downsample= False),
            bottleneckblock(256, 256, downsample= False),
            bottleneckblock(256, 256, downsample= False)
        )

        self.layer2= nn.Sequential(
            bottleneckblock(256, 512, downsample= True),
            bottleneckblock(512, 512, downsample= False),
            bottleneckblock(512, 512, downsample= False),
            bottleneckblock(512, 512, downsample= False)
        )

        self.layer3= nn.Sequential(
            bottleneckblock(512, 1024, downsample= True),
            bottleneckblock(1024, 1024, downsample= False),
            bottleneckblock(1024, 1024, downsample= False),
            bottleneckblock(1024, 1024, downsample= False),
            bottleneckblock(1024, 1024, downsample= False),
            bottleneckblock(1024, 1024, downsample= False)
        )

        self.layer4= nn.Sequential(
            bottleneckblock(1024, 2048, downsample= True),
            bottleneckblock(2048, 2048, downsample= False),
            bottleneckblock(2048, 2048, downsample= False)
        )

        self.final= nn.Sequential(final_layer(2048, num_classes))

    def forward(self, x):
        out = self.final(self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x))))))
        return out
    
