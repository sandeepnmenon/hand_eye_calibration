from .quaternion import Quaternion
import numpy as np
import csv


def read_time_stamped_poses_from_csv_file(csv_file, JPL_quaternion_format=False, time_scale=1.0):
    """
    Reads time stamped poses from a CSV file.
    Assumes the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    The quaternion is expected in Hamilton format, if JPL_quaternion_format is True
    it expects JPL quaternions and they will be converted to Hamiltonian quaternions.
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
    poses = time_stamped_poses[:, 1:]

    quaternions = []
    for pose in poses:
        pose[3:] /= np.linalg.norm(pose[3:])
        if JPL_quaternion_format:
            quaternion_JPL = np.array([-pose[3], -pose[4], -pose[5], pose[6]])
            quaternions.append(Quaternion(q=quaternion_JPL))
        else:
            quaternions.append(Quaternion(q=pose[3:]))

    return (time_stamped_poses.copy(), times, quaternions)


def write_double_numpy_array_to_csv_file(array, csv_file):
    np.savetxt(csv_file, array, delimiter=", ", fmt="%.18f")


def write_time_stamped_poses_to_csv_file(time_stamped_poses, csv_file):
    """
    Writes time stamped poses to a CSV file.
    Uses the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    """
    write_double_numpy_array_to_csv_file(time_stamped_poses, csv_file)


def write_time_stamped_transformation_matrices(time_stamped_poses, file_path):
    """
    Convert time stamped poses to transformation matrices and save them as a numpy file.
    """
    from .pose_evaluation_utils import pose_vec_q_to_mat
    transformation_matrices = []
    for pose in time_stamped_poses:
        transformation_matrices.append(pose_vec_q_to_mat(pose))
    
    np.save(file_path, np.array(transformation_matrices))
