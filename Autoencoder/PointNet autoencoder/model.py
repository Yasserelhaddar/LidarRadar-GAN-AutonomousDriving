import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torchsummary import summary


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], int(x.shape[1] / 2), x.shape[2] * 2)

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], int(x.shape[1] / 2), x.shape[2] * 2)


class PointCloudNet(nn.Module):

    def __init__(self, input_channels=3, output_channels=3, num_points=1024):
        super(PointCloudNet, self).__init__()
        print(num_points)

        self.feature1_module = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.feature2_module = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.feature3_module = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.feature4_module = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU())


        self.UPSAMPLING_module = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1024 + 64 + 256 + 128, out_channels=2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2048, out_channels=2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1),
            nn.BatchNorm1d(3),
            #nn.ReLU(),
            
        )


    def forward(self, pointcloud):

        num_points = pointcloud.shape[1]

        l_feature1 = self.feature1_module(pointcloud.permute(0, 2, 1))
        l_feature2 = self.feature2_module(l_feature1)
        l_feature3 = self.feature3_module(l_feature2)
        l_feature4 = self.feature4_module(l_feature3)

        g_features = nn.MaxPool1d(num_points)(l_feature4)


        c_features = torch.cat([l_feature1, l_feature2, l_feature3, g_features.repeat(1, 1, num_points)], dim=1)

        return self.UPSAMPLING_module(c_features).permute(0, 2, 1)


def test():
    torch.cuda.is_available()
    img_channels = 3
    img_size = 1024
    x = torch.randn((1, 2048*4, 3))
    y = 18 * x
    gen = PointCloudNet()
    print(torch.transpose(gen(y), 1, 2))
    print(torch.transpose(y, 1, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    model = gen.to(device)

    summary(model, (2048*4, 3))


if __name__ == "__main__":
    test()

