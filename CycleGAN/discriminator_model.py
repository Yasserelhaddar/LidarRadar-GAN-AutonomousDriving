import torch
import torch.nn as nn
from torchsummary import summary


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


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels, num_features = 64, features=[64, 128, 256, 512], num_residuals=9):
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
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        x = self.glob_maxpool(x)
        #print(x)
        x = x.reshape(512,-1).repeat(512,1).view(-1)
        x = x.reshape(-1,1,512,512)
        x =self.middle(x)
        #print(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((1, 3, 1024))
    model = Discriminator(1,3,64,[64, 128, 256, 512],2)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 

    device = torch.device(dev)
    
    model = model.to(device)
    
    summary(model,(3, 2^14))


if __name__ == "__main__":

    print(torch.cuda.get_device_name(0))

    test()
