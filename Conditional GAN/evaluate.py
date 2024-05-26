import torch
from dataset import SimRealDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from chamfer_loss import distChamfer
import numpy as np
from plyfile import PlyData, PlyElement
from functions import write_pointcloud

from discriminator import Discriminator
from generator import Generator



def main():
    
    gen = Generator().to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

    #filename = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_train_dataset_2^14\\Sim_dataset\\16384_00001212.ply"
    
    #real_img = PlyData.read(filename)
    #real_img = np.asarray(real_img.elements[0].data)
    #real_img = np.asarray(real_img.tolist())
    #print(real_img)
    #num_points1 = np.random.choice(8192, 2048)
    #real_down = real_img[num_points1, :]

    #real_img = torch.from_numpy(real_img)

    #print(real_img)
    #real_img_transpose = torch.transpose(real_img,1,0).cuda().float()
    #real_img_orig = real_img_transpose.unsqueeze(0).cuda().float()

    noise = torch.randn(config.BATCH_SIZE, 1024, 16384).to(config.DEVICE)
    
    prediction = gen(noise)
    print(prediction)
    #prediction = torch.transpose(prediction,2,1).cuda().float()
    prediction = torch.squeeze(prediction)
    print(prediction.cpu().detach().numpy().shape)

    completeName_ply = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Hello_DCGAN_40.ply'
    #completeName_ply_orig = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Hello_orig_real.ply'
    write_pointcloud(completeName_ply, torch.transpose(prediction,1,0).cuda().float())
    #write_pointcloud(completeName_ply_orig, real_img)
    
if __name__ == "__main__":
    main()
