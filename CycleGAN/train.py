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

from Discriminator import Discriminator
from generator_model2 import Generator

def train_fn(disc_S, disc_R, gen_R, gen_S, loader, opt_disc, opt_gen, mse, d_scaler, g_scaler):
    S_reals = 0
    S_fakes = 0
    loop = tqdm(loader)

    for idx, (sim, real) in enumerate(loop):
        sim = sim.to(config.DEVICE)
        real = real.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_sim = gen_S(real)
            D_S_real = disc_S(sim)
            D_S_fake = disc_S(fake_sim.detach())
            S_reals += D_S_real.mean().item()
            S_fakes += D_S_fake.mean().item()
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            fake_real = gen_R(sim)
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            # put it togethor
            D_loss = (D_S_loss + D_R_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_sim)
            D_R_fake = disc_R(fake_real)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))

            # cycle loss
            cycle_real = gen_R(fake_sim)
            cycle_sim = gen_S(fake_real)
            cycle_real_loss = distChamfer(real, cycle_real)
            cycle_sim_loss = distChamfer(sim, cycle_sim)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_real = gen_R(real)
            identity_sim = gen_S(sim)
            identity_real_loss = distChamfer(real, identity_real)
            identity_sim_loss = distChamfer(sim, identity_sim)

            # add all togethor
            G_loss = (
                loss_G_R
                + loss_G_S
                + cycle_real_loss * config.LAMBDA_CYCLE
                + cycle_sim_loss * config.LAMBDA_CYCLE
                + identity_sim_loss * config.LAMBDA_IDENTITY
                + identity_real_loss * config.LAMBDA_IDENTITY
            )

        if idx % 10 == 0:
            print("Loss GR= ", loss_G_R)
            print("Loss GS= ", loss_G_S)
            print("Loss cycle_real_loss= ", cycle_real_loss)
            print("Loss cycle_sim_loss= ", cycle_sim_loss)
            
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()



def main():
    disc_S = Discriminator(2**14).to(config.DEVICE)
    disc_R = Discriminator(2**14).to(config.DEVICE)
    gen_R = Generator(3, 64 ,2).to(config.DEVICE)
    gen_S = Generator(3, 64 ,2).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    #chamfer = chamfer()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_R, gen_R, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_R, disc_R, opt_disc, config.LEARNING_RATE,
        )

    dataset = SimRealDataset(
        root_sim=config.TRAIN_DIR+"\\Sim_dataset", root_real=config.TRAIN_DIR+"\\Real_dataset")
    val_dataset = SimRealDataset(
       root_sim=config.VAL_DIR+"\\Sim_dataset", root_real=config.VAL_DIR+"\\Real_dataset")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_S, disc_R, gen_R, gen_S, loader, opt_disc, opt_gen, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_R)
            save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_S)
            save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_CRITIC_R)

        print("NUMBER OF EPOCHS: ",epoch)
if __name__ == "__main__":
    main()
