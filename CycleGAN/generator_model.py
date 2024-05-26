import torch
import torch.nn as nn
from torchsummary import summary

from loss import ChamferLoss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            #nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=1),
            #ConvBlock(channels, channels, use_act=False, kernel_size=1),
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
                ConvBlock(num_features*16, num_features*8, down=False, kernel_size=1, stride=1, padding=0, output_padding=0),
                ConvBlock(num_features*8, num_features*4, down=False, kernel_size=1, stride=1, padding=0, output_padding=0),
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=1, stride=1, padding=0, output_padding=0),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=1, stride=1, padding=0, output_padding=0),
            ]
        )

        self.last = nn.Conv1d(num_features*1, img_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        x = self.glob_maxpool(x)
        #print(x)
        x = x.reshape(1024,-1).repeat(1024,1).view(-1)
        x = x.reshape(-1,1024,1024)
        #print(x)
        for layer in self.up_blocks:
            x = layer(x)
            x= nn.Upsample(scale_factor=2, mode='nearest')(x)
            #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        #x= nn.Upsample(scale_factor=2, mode='nearest')(x)
        return self.last(x)






def test():
    torch.cuda.is_available()
    img_channels = 3
    img_size = 1024
    x = torch.randn((1, img_channels, img_size))
    y = 18*x
    gen = Generator(img_channels, 64, 2)
    print(torch.transpose(gen(y), 1, 2))
    print(torch.transpose(y, 1, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    
    model = gen.to(device)
    
    summary(model,(3, 32768))

    #print(ChamferLoss(torch.transpose(gen(y), 1, 2),torch.transpose(y, 1, 2)))

if __name__ == "__main__":
    test()
    m = GlobalMaxPooling1D()
    input = torch.randn(1, 1536, 1024)
    output = m(input)
    print(output.shape)
    print(input.shape)
