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
from Repulsion_loss import knn

import numpy as np

from model import PointCloudNet


def train_fn(gen_S, loader, opt_gen, g_scaler):

    S_reals = 0
    S_fakes = 0
    loop = tqdm(loader)

    for idx, real in enumerate(loop):
        real = real.to(config.DEVICE)

        #print(real.shape)

        num_points1 = np.random.choice(8192, 2048)
        real_down = real[ :, num_points1, :]

        #print(real_down.shape)

        pred = gen_S(real_down)

        Construction_loss_G_S = distChamfer(real, pred)
        Repulsion_loss_G_S = knn(pred)
        G_loss = Construction_loss_G_S+0.1*Repulsion_loss_G_S

        if idx % 1000 == 0:
            print("Loss = ", G_loss)
            print("Construction Loss = ", Construction_loss_G_S)
            print("Repulsion Loss = ", Repulsion_loss_G_S)
            print("Real data = ", real)
            #print("Real data downsampled= ", real_down)
            print("Predicted data = ", pred)
                
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()



def main():
    #gen_S = Generator(img_channels=3, num_features=64, num_residuals=0).to(config.DEVICE)
    gen_S = PointCloudNet().to(config.DEVICE)
    
    opt_gen = optim.Adam(gen_S.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    #opt_gen = optim.SGD(gen_S.parameters(), lr=0.00001, momentum=0.9)



    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE,
        )
        

    dataset = SimRealDataset(config.TRAIN_DIR)
    #val_dataset = SimRealDataset(config.VAL_DIR)
    #val_loader = DataLoader(
        #val_dataset,
        #batch_size=1,
        #shuffle=False,
        #pin_memory=False)
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    g_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(gen_S, loader, opt_gen, g_scaler)

        print(epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)


if __name__ == "__main__":
    main()
