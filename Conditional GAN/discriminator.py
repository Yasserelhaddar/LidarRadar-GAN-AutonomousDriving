import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from loss import ChamferLoss
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, padding_mode="reflect"),
            nn.BatchNorm1d(channels),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, num_points):
        super(Encoder, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.residual = ResidualBlock(1024)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024 + 64, 1024)
        self.fc2 = nn.Linear(1024, 512)  # codeword dimension = 512
        #self.bn4 = nn.BatchNorm1d(1024)
        #self.bn5 = nn.BatchNorm1d(512)

    def forward(self, input):
        #input = input.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(input)))
        local_feature = x  # save the  low level features to concatenate this global feature.
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.residual(x)
        x = self.residual(x)
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        feature = torch.cat([local_feature, global_feature], 1)  # [bs, 1088, 2048]

        # TODO: add batch_norm or not?
        x = F.relu(self.fc1(feature.transpose(1, 2)))
        x = F.relu(self.fc2(x))

        # TODO: the actual output should be [bs, 1, 512] by max pooling ??
        return torch.max(x, 1, keepdim=True)[0]  # [bs, 1, 512]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



class PatchGAN(nn.Module):
    def __init__(self, in_channels, img_channels, features=[64, 128, 256, 512]):
        super(PatchGAN, self).__init__()
        self.middle = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(512,-1).repeat(512,1).view(-1)
        x = x.reshape(-1,1,512,512)
        x = self.middle(x)
        return torch.sigmoid(self.model(x))




class Discriminator(nn.Module):
    def __init__(self, num_points):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(num_points=num_points)
        self.patchGAN = PatchGAN(1, 3, [64, 128, 256, 512])

    def forward(self, input):
        codeword = self.encoder(input)
        output = self.patchGAN(codeword)
        return output
    
    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.patchGAN.parameters())



def test():
    torch.cuda.is_available()
    img_channels = 3
    img_size = 1024
    x = torch.randn((1, 2**14, 3))
    y = 18*x
    y = y.cuda()
    disc = Discriminator(2**14)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = disc.to(device)
    
    summary(model,(3, 2**14))
    #print(model.get_loss(y,model(y)))
    #print(y)
    #print(model(y))
    
if __name__ == "__main__":

    test()

