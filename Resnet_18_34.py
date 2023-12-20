# This .py file will be used to implement the basic Blocks for Resnt as well as
# the architecture as presented in the following paper: https://arxiv.org/abs/1512.03385

# Important reminder regarding output shape of a Conv2D layer, assuming we have:
# input spatial size of W, stride of S, padding of P and filter (kernel) size of F 
# we can calculate the output shape as floor((W - F + 2P)/S + 1)

# Importing the relevant packages
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torchsummary import summary


# Defining the basic block of Resnet with downsample option:      
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        # Using of stride=2 in the first part allows us to downsample because:
        #  outshape= floor((W - 3 + 2)/2 + 1) = W/2
        #  outshape= floor((W - 1 + 0)/2 +1) = W/2
        if downsample == True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride=2, padding=1)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.skip = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Initialization
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if len(self.skip) > 0:
            init.kaiming_normal_(self.skip[0].weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        skip = self.skip(x)
        var1 = F.relu(self.bn1(self.conv1(x)))
        var2 = F.relu(self.bn2(self.conv2(var1)))
        out = var2 + skip
        return F.relu(out)

# Defining layer_0:
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

# Defining Final layer:
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
    

# Putting it all together for building Resnet18:
class ResNet18(nn.Module):
    def __init__(self, in_channels, layer0, resblock, final_layer, num_classes):
        super().__init__()

        self.layer0= nn.Sequential(layer0(in_channels))

        self.layer1= nn.Sequential(
            resblock(64, 64, downsample= False),
            resblock(64, 64, downsample= False)
        )

        self.layer2= nn.Sequential(
            resblock(64, 128, downsample= True),
            resblock(128, 128, downsample= False)
        )

        self.layer3= nn.Sequential(
            resblock(128, 256, downsample= True),
            resblock(256, 256, downsample= False)
        )

        self.layer4= nn.Sequential(
            resblock(256, 512, downsample= True),
            resblock(512, 512, downsample= False)
        )

        self.final= nn.Sequential(final_layer(512, num_classes))

    def forward(self, x):
        out = self.final(self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x))))))
        return out


# Using resblock to build Resnet34
class ResNet34(nn.Module):
    def __init__(self, in_channels, layer0, resblock, final_layer, num_classes=1000):
        super().__init__()

        self.layer0= nn.Sequential(layer0(in_channels))

        self.layer1= nn.Sequential(
            resblock(64, 64, downsample= False),
            resblock(64, 64, downsample= False),
            resblock(64, 64, downsample= False)
        )

        self.layer2= nn.Sequential(
            resblock(64, 128, downsample= True),
            resblock(128, 128, downsample= False),
            resblock(128, 128, downsample= False),
            resblock(128, 128, downsample= False)
        )

        self.layer3= nn.Sequential(
            resblock(128, 256, downsample= True),
            resblock(256, 256, downsample= False),
            resblock(256, 256, downsample= False),
            resblock(256, 256, downsample= False),
            resblock(256, 256, downsample= False),
            resblock(256, 256, downsample= False)
        )

        self.layer4= nn.Sequential(
            resblock(256, 512, downsample= True),
            resblock(512, 512, downsample= False),
            resblock(512, 512, downsample= False)
        )

        self.final= nn.Sequential(final_layer(512, num_classes))

    def forward(self, x):
        out = self.final(self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x))))))
        return out
    