import torch
import torch.nn as nn
from torchsummary import summary

from chamfer_loss import distChamfer

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


class Generator(nn.Module):

    def __init__(self, input_channels=3, output_channels=3, embed_size):
        super(Generator, self).__init__()
        #print(num_points)

        self.residual = ResidualBlock(1024)

	self.embed = nn.Embedding(num_classes, embed_size)


        self.UPSAMPLING_module = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1024+embed_size, out_channels=2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2048, out_channels=2048, kernel_size=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #self.residual,
            #self.residual,
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


    def forward(self, codeword, labels):

	embedding = self.embed(labels).view(labels.shape[0], 1024, self.num_points)

	x = torch.cat([codeword, embedding], dim=1)

        return self.UPSAMPLING_module(x)






def test():
    torch.cuda.is_available()
    x = torch.randn((1,1024,16384))
    y = 18*x
    gen = Generator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = gen.to(device)

    print(model(y.cuda()))
    print(y.cuda())
    
    
    summary(model,(1024,16384))

    #print(distChamfer(y.cuda(),model(y.cuda())))

if __name__ == "__main__":
    test()

