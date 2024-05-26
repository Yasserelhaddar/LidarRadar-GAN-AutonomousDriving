import os
import numpy as np
from functions import pcl2voxel, voxel2pcl, write_pointcloud
from plyfile import PlyData, PlyElement

directory = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_train_dataset\\Sim_dataset'
#directory = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\data_8192\\Sim_data'

directory1 = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_train_dataset\\Real_dataset'
#directory1 = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\data_8192\\Real_data'

#voxels = []
for filename in os.listdir(directory):
    if filename.endswith(".ply") :
        directory_file = os.path.join(directory, filename)
        print(directory_file)
        plydata = PlyData.read(directory_file)
        data = plydata.elements[0].data
        x =[]
        y =[]
        z =[]
        for i in range(0, data.size-1):
            x.append(data['x'][i])
            y.append(data['y'][i])
            z.append(data['z'][i])
            #intensity.append(data['I'][i])

        points = []
        for i, j, k in zip(x, y, z):
            if -40 <= i <= 40 and -40 <= j <= 40:
                points.append((i,j,k))

        points = np.asarray(points)
        points = points.reshape(-1,3)

        sim_img = np.asarray(points)
        sim_img = np.asarray(sim_img.tolist())
        if sim_img.shape[0] < 2**14:
            M = np.abs(2**14 - sim_img.shape[0])
            sim_img = np.pad(sim_img, ((0, M), (0, 0)), 'symmetric')
        else:
            num_points1 = np.random.choice(sim_img.shape[0], 2**14)
            sim_img = sim_img[num_points1]

        
        #voxel, segments, shape, n = pcl2voxel(data, 64)
        #voxels.append(voxel)
        
        #Convert voxel file on a voxelized grid into PCL.ply file
        #voxel2pcl(filename, voxel, segments, shape, n)

        completeName = 'C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/CycleGan_train_dataset_2^14/Sim_dataset'
        completeName_ply = os.path.join(completeName,"16384_"+filename)

        PCL_out_array = write_pointcloud(completeName_ply,sim_img)

#np.savez_compressed('C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/New_data/Sim_data_16384.npz', a = np.asarray(voxels))

#voxels = []

for filename in os.listdir(directory1):
    if filename.endswith(".ply") :
        directory_file = os.path.join(directory1, filename)
        print(directory_file)
        plydata = PlyData.read(directory_file)
        data = plydata.elements[0].data
        x =[]
        y =[]
        z =[]
        for i in range(0, data.size-1):
            x.append(data['x'][i])
            y.append(data['y'][i])
            z.append(data['z'][i])
            #intensity.append(data['I'][i])

        points = []
        for i, j, k in zip(x, y, z):
            if -40 <= i <= 40 and -40 <= j <= 40:
                points.append((i,j,k))

        points = np.asarray(points)
        points = points.reshape(-1,3)

        sim_img = np.asarray(points)
        sim_img = np.asarray(sim_img.tolist())
        if sim_img.shape[0] < 2**14:
            M = np.abs(2**14 - sim_img.shape[0])
            sim_img = np.pad(sim_img, ((0, M), (0, 0)), 'symmetric')
        else:
            num_points1 = np.random.choice(sim_img.shape[0], 2**14)
            sim_img = sim_img[num_points1]

        
        #voxel, segments, shape, n = pcl2voxel(data, 64)
        #voxels.append(voxel)
        #Convert voxel file on a voxelized grid into PCL.ply file
        #voxel2pcl(filename, voxel, segments, shape, n)

        completeName = 'C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/CycleGan_train_dataset_2^14/Real_dataset'
        completeName_ply = os.path.join(completeName,"16384_"+filename)

        PCL_out_array = write_pointcloud(completeName_ply,sim_img)
        
#np.savez_compressed('C:/Users/ElHaddar/OneDrive/Desktop/Cycle GAN/New_data/Real_data_16384.npz', a = np.asarray(voxels))
