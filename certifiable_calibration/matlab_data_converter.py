"""
Converts transformation matrices to rotation and translation vectors as required by the matlab solver in certifiable_calibration module
"""

import numpy as np
from extrinsic_calibration.solver.ext_solver import solver
from extrinsic_calibration.utility.utils import load_data_1, inertial_to_relative, add_noise, load_data_from_numpy, inertial_to_relative_numpy
import math
import argparse
import scipy.io as sio

def rotation_matrix_to_euler_angles(R):
    """
    converts rotation matrix to euler angles.
    @param R: 3 * 3 rotation matrix
    @return: euler angles roll,pitch and yaw in degrees.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def get_extrinsic_params_from_transformation_matrix(transformation_matrix):
    [px, py, pz] = [transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]]
    [roll, pitch, yaw] = rotation_matrix_to_euler_angles(transformation_matrix[np.ix_([0, 1, 2], [0, 1, 2])])
    return {
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'px': px,
        'py': py,
        'pz': pz,
    }

def get_rotation_translation_from_transformation_matrix(transformation_matrix):
    [px, py, pz] = [transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]]

    return np.array([px, py, pz]), transformation_matrix[np.ix_([0, 1, 2], [0, 1, 2])]

# Name the relevant filenames
# data_filename = "unscaled_pose_data.mat"
T_vki_filename = "/home/menonsandu/Downloads/MH_01_easy/mav0/state_groundtruth_estimate0/data_1_aligned.npy"
T_cki_filename = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/kf_MH01_full_aligned.npy"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize IMU and camera trajectory.')
    parser.add_argument('--T_vki_filename', type=str, default=T_vki_filename, help='IMU trajectory filename.')
    parser.add_argument('--T_cki_filename', type=str, default=T_cki_filename, help='Camera trajectory filename.')
    parser.add_argument('--output_filename', type=str, default="trajectory_data.mat", help='Output filename.')
    args = parser.parse_args()

    # Check if output filename does not end with .mat
    if args.output_filename[-4:] != ".mat":
        args.output_filename += ".mat"

    # Load the data
    # T_vki, T_cki, extcal = load_data_1(data_filename)
    T_vki = np.load(args.T_vki_filename)
    T_cki = np.load(args.T_cki_filename)
    print(len(T_vki), len(T_cki))

    scale = 1.0

    #Set noise levels
    trans_noise = 0.01 #Percentage of translation
    rot_noise = 0.01 #Percentage of rotation

    # Convert inertial poses to relative
    T_v_rel = inertial_to_relative_numpy(T_vki)
    T_c_rel = inertial_to_relative_numpy(T_cki)
    # T_v_rel = T_vki
    # T_c_rel = T_cki

    R1 =[]
    R2 = []
    t1 = []
    t2 = []
    for T_v, T_c in zip(T_v_rel, T_c_rel):
        # print(T_v, T_c)
        translation1, rotation1 = get_rotation_translation_from_transformation_matrix(np.array(T_v))
        translation2, rotation2 = get_rotation_translation_from_transformation_matrix(np.array(T_c))
        R1.append(rotation1)
        R2.append(rotation2)
        t1.append(translation1)
        t2.append(translation2)

        # break
    R1 = np.transpose(R1, (1,2,0))
    R2 = np.transpose(R2, (1,2,0))
    t1 = np.transpose(t1, (1,0))
    t2 = np.transpose(t2, (1,0))


    matlab_data = {"R1": R1, "R2": R2, "t1" : t1, "t2": t2}
    # print(matlab_data)
    sio.savemat(args.output_filename, matlab_data)

 # 168.5032   -2.9494   51.8078
 # 110.7732  -65.1899  173.2599
 # -115.3655  -23.1548   53.6081
 #  72.0353   58.3147 -103.1728
