import torch
from dataset import SimRealDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_CGAN
from tqdm import tqdm
from chamfer_loss import distChamfer

from discriminator_CGAN import Discriminator
from generator_CGAN import Generator

def train_fn(disc, gen, loader, opt_disc, opt_gen, criterion):
    loop = tqdm(loader)
    L_total_gen = []
    L_total_disc = []
    for batch_idx, (real, labels) in enumerate(loop):
	labels = labels.to(config.DEVICE)
        real = real.to(config.DEVICE)
        noise = torch.randn(config.BATCH_SIZE, 1024, 16384).to(config.DEVICE)
        fake = gen(noise, labels)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real, labels).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach(), labels).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake, labels).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx%1000 == 0:
            print("Loss disc : ", loss_disc)
            print("Loss gen : ", loss_gen)
            print("Real : ", real)
            print("fake : ", fake)




def main():
    disc = Discriminator(2**14, config.NUM_CLASSES).to(config.DEVICE)
    gen = Generator(config.NUM_CLASSES, config.GEN_EMBEDDING).to(config.DEVICE)


    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    dataset = SimRealDataset(root_real=config.TRAIN_DIR+"\\Real_dataset", root_labels=config.TRAIN_DIR_LABELS)
   
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, loader, opt_disc, opt_gen, criterion)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        print("NUMBER OF EPOCHS: ",epoch)
        
if __name__ == "__main__":
    main()
