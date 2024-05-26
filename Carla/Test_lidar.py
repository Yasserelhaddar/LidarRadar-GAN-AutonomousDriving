import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

im_width = 640
im_height = 480


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((im_height, im_width, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    # --------------
    # Add a new LIDAR sensor to my ego
    # --------------
    lidar_cam = None
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',str(64))
    lidar_bp.set_attribute('points_per_second',str(10000000))
    lidar_bp.set_attribute('rotation_frequency',str(100))
    lidar_bp.set_attribute('range',str(120))
    lidar_location = carla.Location(0,0,2)
    lidar_rotation = carla.Rotation(0,0,0)
    lidar_transform = carla.Transform(lidar_location,lidar_rotation)
    lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle)

    
    # add sensor to list of actors
    actor_list.append(lidar_sen)

    lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('C:/Users/49152/Desktop/Dataset/Simulated data/%.11d.ply' % point_cloud.frame))


    # sleep for 200 seconds, then finish:
    time.sleep(2500)

finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
