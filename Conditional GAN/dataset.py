import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData, PlyElement
import config

class SimRealDataset(Dataset):
    def __init__(self, root_real):
        self.root_real = root_real
        self.real_images = os.listdir(root_real)
        self.length_dataset = len(self.real_images) # 1000, 1500
        self.real_len = len(self.real_images)

    def __len__(self):

        return self.length_dataset

    def __getitem__(self, index): ################################################## IMPORTANT ##########
        real_img = self.real_images[index % self.real_len]


        real_path = os.path.join(self.root_real,real_img)

        
        real_img = PlyData.read(real_path)
        real_img = np.asarray(real_img.elements[0].data)
        real_img = np.asarray(real_img.tolist())
        '''
        real_img = np.asarray(real_img.tolist())
        if real_img.shape[0] < 32768:
            N = np.abs(32768 - real_img.shape[0])
            real_img = np.pad(real_img, ((0, N), (0, 0)), 'symmetric')
        else:
            num_points2 = np.random.choice(real_img.shape[0], 32768)
            real_img = real_img[num_points2]

        #real_img.resize(32768, 3)
        '''
        real_img = torch.from_numpy(real_img)
        #real_img = real_img.cuda().float()
        
        real_img = torch.transpose(real_img,1,0).cuda().float()
        #real_img = torch.transpose(real_img,1,2).cuda().float()

        #real_img = real_img.unsqueeze(0).cuda().float()

        #print(sim_img)

        return real_img
    ##########################################################################################################

'''
def test():
    dataset = SimRealDataset(root_sim=config.TRAIN_DIR + "\\Sim_dataset", root_real=config.TRAIN_DIR + "\\Real_dataset")
    gen = Generator(3, 64, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    model = gen.to(device)
    for i in range(1):
        #dataset[i]
        sample_real, sample_sim = dataset[i]
        print(model(sample_real))
        print(sample_real.shape)
        print(distChamfer(model(sample_real),sample_real))






if __name__ == "__main__":

    print(torch.cuda.get_device_name(0))

    test()
'''

