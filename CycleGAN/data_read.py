from functions import write_pointcloud
import numpy as np
from plyfile import PlyData, PlyElement
import os

#filename = 'C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/CycleGan_test_dataset/Sim_dataset/00001490.ply'

directory = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_test_dataset_2^15\\Real_dataset'

for filename in os.listdir(directory):
    if filename.endswith(".ply") :
        directory_file = os.path.join(directory, filename)
        file = PlyData.read(directory_file)
        data = np.asarray(file.elements[0].data)
        data2 = np.asarray(data.tolist())
        print(data)
        print(data2)

        '''
        print(directory_file)

        final_data = []
        for k in data:
            x = k[0]
            y = k[1]
            z = k[2]
            new_data = [x,y,z]
            final_data.append(new_data)

        final_data = np.asarray(final_data)

        sim_img = np.asarray(final_data)
        sim_img = np.asarray(sim_img.tolist())
        if sim_img.shape[0] < 65536:
            M = np.abs(65536 - sim_img.shape[0])
            sim_img = np.pad(sim_img, ((0, M), (0, 0)), 'symmetric')
        else:
            num_points1 = np.random.choice(sim_img.shape[0], 65536)
            sim_img = sim_img[num_points1]

        completeName_ply = 'C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/CycleGan_test_dataset_2^16/Real_dataset/_65536_'+str(filename)
        write_pointcloud(completeName_ply, sim_img)

        sim_img = np.asarray(final_data)
        sim_img = np.asarray(sim_img.tolist())
        if sim_img.shape[0] < 32768:
            M = np.abs(32768 - sim_img.shape[0])
            sim_img = np.pad(sim_img, ((0, M), (0, 0)), 'symmetric')
        else:
            num_points1 = np.random.choice(sim_img.shape[0], 32768)
            sim_img = sim_img[num_points1]

        completeName_ply = 'C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/CycleGan_test_dataset_2^15/Real_dataset/_32768_'+str(filename)
        write_pointcloud(completeName_ply, sim_img)'''
