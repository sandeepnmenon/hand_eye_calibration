import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

import pykitti

from pose_alignment_module.pose_evaluation_utils import pose_mat_to_vec_q

def load_timestamps(timestamp_file):
    """Load timestamps from file."""

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(t)
    
    return timestamps

def odometry_pose_to_quat(basedir, sequence):
    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.odometry(basedir, sequence)

    poses = dataset.poses
    timestamps = dataset.timestamps

    quat_poses = [pose_mat_to_vec_q(pose) for pose in poses]
    timestamps = [timestamp.total_seconds() for timestamp in timestamps]
    # replace first element in quat_poses with timestamps
    for i in range(len(quat_poses)):
        quat_poses[i][0] = timestamps[i]

    pose_file = os.path.join(dataset.pose_path, dataset.sequence + '.txt')
    output_pose_file = os.path.join(dataset.pose_path, dataset.sequence + '_quat.csv')

    # Write the quaternion poses to the output file
    with open(output_pose_file, 'w') as f:
        for pose in quat_poses:
            f.write(','.join(map(str, pose)) + '\n')

    print(f"Wrote quaternion poses to {output_pose_file}")


def raw_oxts_pose_to_quat(basedir, date, sequence):
    dataset = pykitti.raw(basedir, date, sequence)

    poses = [oxts_data.T_w_imu for oxts_data in dataset.oxts]
    quat_poses = [pose_mat_to_vec_q(pose) for pose in poses]
    timestamps = [timestamp.timestamp() for timestamp in dataset.timestamps]
    # replace first element in quat_poses with timestamps
    for i in range(len(quat_poses)):
        quat_poses[i][0] = timestamps[i]

    output_pose_file = os.path.join(dataset.data_path, 'oxts', 'quat_poses.csv')
    # Write the quaternion poses to the output file
    with open(output_pose_file, 'w') as f:
        for pose in quat_poses:
            f.write(','.join(map(str, pose)) + '\n')

    print(f"Wrote quaternion poses to {output_pose_file}")

def raw_oxts_timestamp_to_seconds(basedir, date, sequence):
    dataset = pykitti.raw(basedir, date, sequence)

    timestamps = [timestamp.timestamp() for timestamp in dataset.timestamps]
    output_file = os.path.join(dataset.data_path, 'oxts', 'times.txt')
    
    # Write new timestamps to output_file
    with open(output_file, 'w') as f:
        for timestamp in timestamps:
            f.write(str(timestamp) + '\n')

def raw_image_timestamp_to_seconds(basedir, date, sequence, image_name):
    dataset = pykitti.raw(basedir, date, sequence)

    timestamp_file = os.path.join(dataset.data_path, image_name, 'timestamps.txt')
    timestamps = load_timestamps(timestamp_file)
    timestamps = [timestamp.timestamp() for timestamp in timestamps]

    output_file = os.path.join(dataset.data_path, image_name, 'times.txt')
    
    # Write new timestamps to output_file
    with open(output_file, 'w') as f:
        for timestamp in timestamps:
            f.write(str(timestamp) + '\n')


if __name__ == '__main__':
    # odometry_pose_to_quat(basedir='/home/menonsandu/Documents/KITTI/odometry/dataset', sequence='03')
    # raw_oxts_timestamp_to_seconds(basedir='/media/menonsandu/Chest/Ubuntu/Downloads/KITTI_2011_09_26_drive_0005_sync', date='2011_09_26', sequence='0005')
    # raw_image_timestamp_to_seconds(basedir='/media/menonsandu/Chest/Ubuntu/Downloads/KITTI_2011_09_26_drive_0005_sync', date='2011_09_26', sequence='0005', image_name='image_00')
    raw_oxts_pose_to_quat(basedir='/media/menonsandu/Chest/Ubuntu/Downloads/KITTI_2011_09_26_drive_0005_sync', date='2011_09_26', sequence='0005')