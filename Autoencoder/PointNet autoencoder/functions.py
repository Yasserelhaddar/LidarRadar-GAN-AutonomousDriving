import os
import numpy as np

import pandas as pd
import scipy.ndimage

import struct

from plyfile import PlyData, PlyElement


def write_pointcloud(filename,xyz_points):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fff",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2])))
    fid.close()

def voxel2pcl(input_array, filename, voxel, segments, shape, n):
    
    PCL_out=[]

        
    for x in range(0,n-1):
        for y in range(0,n-1):
            for z in range(0,n-1):
                if voxel [x][y][z] == True:
                    xc= segments[0][x]+shape[0]/2
                    yc= segments[1][y]+shape[1]/2
                    zc= segments[2][z]+shape[2]/2
                    PCL_out.append([xc,yc,zc])


    if input_array:
        completeName = os.path.join(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\Results_VoxelNet_32',"Input_32"+filename+".txt")
        completeName_ply = os.path.join(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\Results_VoxelNet_32',"Input_32"+filename+".ply")
    else:
        completeName = os.path.join(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\Results_VoxelNet_32',"Prediction_32"+filename+".txt")
        completeName_ply = os.path.join(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\Results_VoxelNet_32',"Prediction_32"+filename+".ply")
            
    with open(completeName, "w") as output:
        for i in range(0, len(PCL_out)-1):
            output.write(str(PCL_out[i]))


    PCL_out_array = write_pointcloud(completeName_ply,np.asarray(PCL_out))


def pcl2voxel(data, n):
    
    x_y_z = np.asarray([n, n, n])
    regular_bounding_box = True

    pcl_id = None
    xyzmin, xyzmax = None, None
    segments = None
    shape = None
    n_voxels = None
    voxel_x, voxel_y, voxel_z = None, None, None
    voxel_n = None
    voxel_centers = None

    points = []

    x = []
    y = []
    z = []
     
    intensity = []



    for i in range(0, data.size-1):
        x.append(data['x'][i])
        y.append(data['y'][i])
        z.append(data['z'][i])
        #intensity.append(data['I'][i])

    for i, j, k in zip(x, y, z):
        points.append((i,j,k))

    points = np.asarray(points)
    points = points.reshape(-1,3)


    xyzmin = points.min(0)
    xyzmax = points.max(0)
    xyz_range = points.ptp(0)

    if regular_bounding_box:
        #: adjust to obtain a minimum bounding box with all sides of equal length
        margin = max(xyz_range) - xyz_range
        xyzmin = xyzmin - margin / 2
        xyzmax = xyzmax + margin / 2


    segments = []
    shape = []
    for i in range(3):
        # note the +1 in num
        s, step = np.linspace(xyzmin[i],xyzmax[i],num=(x_y_z[i] + 1),retstep=True)
        segments.append(s)
        shape.append(step)

    n_voxels = np.prod(x_y_z)


    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based; clip for edge cases
    voxel_x = np.clip(np.searchsorted(segments[0], points[:, 0]) - 1, 0, x_y_z[0])
    voxel_y = np.clip(np.searchsorted(segments[1], points[:, 1]) - 1, 0, x_y_z[1])
    voxel_z = np.clip(np.searchsorted(segments[2], points[:, 2]) - 1, 0,  x_y_z[2])
    voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], x_y_z)


    # compute center of each voxel
    midsegments = [(segments[i][1:] + segments[i][:-1]) / 2 for i in range(3)]

    voxel = np.zeros((n, n, n)).astype(np.int8)

    for x, y, z in zip(voxel_x, voxel_y, voxel_z):
        voxel[x][y][z] = 1

    return voxel, segments, shape, n

   
