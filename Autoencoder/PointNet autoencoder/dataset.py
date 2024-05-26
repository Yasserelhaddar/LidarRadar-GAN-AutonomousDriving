import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData, PlyElement
import config
from chamfer_loss import distChamfer

class SimRealDataset(Dataset):
    def __init__(self, root_real):
        self.root_real = root_real
        self.real_images = os.listdir(root_real)

        self.length_dataset = len(self.real_images)
        self.real_len = len(self.real_images)


    def __len__(self):

        return self.length_dataset

    def __getitem__(self, index): ################################################## IMPORTANT ##########
        real_img = self.real_images[index % self.real_len]


        real_path = os.path.join(self.root_real,real_img)
        
        real_img = PlyData.read(real_path)
        real_img = np.asarray(real_img.elements[0].data)
        real_img = np.asarray(real_img.tolist())

        real_img = torch.from_numpy(real_img).cuda().float()
        #real_img = torch.transpose(real_img,1,0).cuda().float()

        #real_img = real_img.unsqueeze(0).cuda().float()

        #print(sim_img)

        return real_img
    ##########################################################################################################
