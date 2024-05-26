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

from model import PointCloudNet



def main():
    
    gen_S = PointCloudNet().to(config.DEVICE)
    opt_gen = optim.Adam(list(gen_S.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE)

    filename = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\8192_Data_lidar_FORD819.ply"

    real_img = PlyData.read(filename)
    real_img = np.asarray(real_img.elements[0].data)
    real_img = np.asarray(real_img.tolist())
    print(real_img)
    num_points1 = np.random.choice(8192, 2048)
    real_down = real_img[num_points1, :]

    real_img = torch.from_numpy(real_down)

    print(real_img)
    #real_img_transpose = torch.transpose(real_img,1,0).cuda().float()
    real_img_orig = real_img.unsqueeze(0).cuda().float()

    prediction = gen_S(real_img_orig)
    print(prediction)
    #prediction = torch.transpose(prediction,2,1).cuda().float()
    prediction = torch.squeeze(prediction)
    print(prediction.cpu().detach().numpy().shape)

    completeName_ply = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Hello.ply'
    completeName_ply_orig = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Hello_orig.ply'
    write_pointcloud(completeName_ply, prediction)
    write_pointcloud(completeName_ply_orig, real_img)
    
if __name__ == "__main__":
    main()
