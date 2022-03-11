import numpy as np
import pandas as pd
import csv
import argparse
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def read_time_stamped_poses_from_csv_file(csv_file,  time_scale=1.0):
    """
    Reads time stamped poses from a CSV file.
    Assumes the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    """
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        csv_data = list(csv_reader)
        if 'time' in csv_data[0][0]:
            # The first line is the header
            csv_data = csv_data[1:]
        time_stamped_poses = np.array(csv_data)
        time_stamped_poses = time_stamped_poses.astype(float)

    time_stamped_poses[:, 0] *= time_scale
    # Extract the quaternions from the poses.
    times = time_stamped_poses[:, 0].copy()
    xyz = time_stamped_poses[:, 1:4].copy()
    quaternions = time_stamped_poses[:, 4:8].copy()
    
    return times, xyz, quaternions

# Default values for local testing
imu_trajectory_filename = "/home/menonsandu/Downloads/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
camera_trajectory_filename = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Stereo/CameraTrajectory.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize IMU and camera trajectory.')
    parser.add_argument('--imu_trajectory_filename', type=str, default=imu_trajectory_filename, help='IMU trajectory filename.')
    parser.add_argument('--camera_trajectory_filename', type=str, default=camera_trajectory_filename, help='Camera trajectory filename.')
    args = parser.parse_args()

    _, imu_xyz, imu_quaternions = read_time_stamped_poses_from_csv_file(args.imu_trajectory_filename)                
    _, camera_xyz, camera_quaternions = read_time_stamped_poses_from_csv_file(args.camera_trajectory_filename)

    # Plot IMU and Camera trajectory using xyz values
    f = pyplot.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlabel('x ')
    ax.set_ylabel('y ')
    ax.set_zlabel('z ')

    ax.plot(imu_xyz[:,0],imu_xyz[:,1],imu_xyz[:,2], label='IMU', color='blue')
    ax.plot(camera_xyz[:,0],camera_xyz[:,1],camera_xyz[:,2], label='Camera', color='red')
    pyplot.show()