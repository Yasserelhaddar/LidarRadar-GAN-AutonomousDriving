import glob
import os
import sys
import math
import numpy as np

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
    return i3 / 255.0


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
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    # --------------
    # Add a new radar sensor to my ego
    # --------------
    rad_cam = None
    rad_bp = world.get_blueprint_library().find('sensor.other.radar')
    rad_bp.set_attribute('horizontal_fov', str(35))
    rad_bp.set_attribute('vertical_fov', str(45))
    rad_bp.set_attribute('range', str(120))
    #rad_bp.set_attribute('channels', str(64))
    rad_bp.set_attribute('points_per_second', str(100000))
    rad_location = carla.Location(x=2.0, z=1.0)
    rad_rotation = carla.Rotation(pitch=5)
    rad_transform = carla.Transform(rad_location, rad_rotation)
    rad_ego = world.spawn_actor(rad_bp, rad_transform, attach_to=vehicle,
                                attachment_type=carla.AttachmentType.Rigid)


    def rad_callback(radar_data):
        velocity_range = 7.5  # m/s
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            world.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
            print(detect)

    def save_data_ply(radar_data):

        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))
        azimuth_array = points[1][:]
        altitude_array = points[2][:]
        depth_array = points[3][:]
        print(points)
        np.savetxt('C:/Users/49152/Desktop/Dataset/Simulated data/%.11d.txt' % radar_data.frame,points)



    #rad_ego.listen(lambda radar_data: rad_callback(radar_data))

    rad_ego.listen(lambda radar_data: save_data_ply(radar_data))


    #rad_ego.listen(lambda point_cloud: point_cloud.save_to_disk('C:/Users/49152/Desktop/Dataset/Simulated data/%.11d.ply' % point_cloud.frame))

    # sleep for 200 seconds, then finish:
    time.sleep(2500)

finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
