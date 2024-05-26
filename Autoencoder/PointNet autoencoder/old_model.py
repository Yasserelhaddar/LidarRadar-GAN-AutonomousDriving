import torch
import torch.nn as nn
from torchsummary import summary
from chamfer_loss import distChamfer
import numpy as np

from model1 import FoldNet

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) 

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            #ConvBlock(channels, channels, kernel_size=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)
    
    
class GlobalMaxPooling1D(nn.Module):

    def __init__(self, data_format=2):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values


class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(img_channels, num_features, kernel_size=7, stride=2, padding=3, padding_mode="reflect"),
            #nn.InstanceNorm1d(num_features),
            #nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*4, num_features*8, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*8, num_features*16, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*16) for _ in range(num_residuals)]
        )
        
        self.glob_maxpool = GlobalMaxPooling1D()
        
        self.up_blocks = nn.ModuleList(
            [
                nn.Conv2d(1, 4, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(1024, 512, kernel_size=(3,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 512, kernel_size=(3,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 128, kernel_size=(3,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 4, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(4, 1, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
                #nn.Upsample(scale_factor=(1,2), mode='nearest'),
                #nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                #nn.Upsample(scale_factor=(1,2), mode='nearest')

            ]
        )

        self.last = nn.Conv1d(num_features*1, img_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        global_feature = self.glob_maxpool(x)
        #print(x)
        x = global_feature.reshape(1024,-1).repeat(32768,1).view(-1)
        x = x.reshape(-1,1,1024,32768)
        #x = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))(x)
        #x = nn.Upsample(scale_factor=(2,1), mode='nearest')(x)
        #print(x)
        for layer in self.up_blocks:
            x = layer(x)
            #x= nn.Upsample(scale_factor=(2,2), mode='nearest')(x)
            #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #return self.last(x)
        return x

        #return global_feature






def test():


    
    torch.cuda.is_available()
    img_channels = 3
    img_size = 1024
    x = torch.randn((1, img_channels, 32768))
    y = 18*x
    y = y.cuda()
    gen = Generator(img_channels, 64, 0)
    model1 = FoldNet(32768)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = gen.to(device)
    model2 = model1.to(device)
    
    summary(model,(3, 32768))
    
    #print(np.min(np.asarray(model(y).cpu().detach().numpy())))
    print(model(y))
    print(model2(torch.transpose(y, 1, 2)))                   
    print(torch.transpose(y, 1, 2))

    #print(distChamfer(torch.transpose(y, 1, 2), torch.transpose(model(y), 1, 2)))

    


if __name__ == "__main__":

    test()


