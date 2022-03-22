import numpy as np
from certifiable_calibration.extrinsic_calibration.solver.ext_solver import solver
from certifiable_calibration.extrinsic_calibration.utility.utils import load_data_1, inertial_to_relative, add_noise, load_data_from_numpy
import math

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

def get_extrinsic_params_from_transformation_matrix(transformation_matrix, scale=1.0):
    [px, py, pz] = [transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]]
    [roll, pitch, yaw] = rotation_matrix_to_euler_angles(transformation_matrix[np.ix_([0, 1, 2], [0, 1, 2])])
    return {
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'px': px*scale,
        'py': py*scale,
        'pz': pz*scale,
    }

# Name the relevant filenames
T_vki_filename = "/home/menonsandu/Downloads/MH_01_easy/mav0/state_groundtruth_estimate0/data_1_aligned.npy"
T_cki_filename = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/kf_MH01_500_aligned.npy"


# Load the data
T_vki = load_data_from_numpy(T_vki_filename)
T_cki = load_data_from_numpy(T_cki_filename)
print(len(T_vki), len(T_cki))
# if len(T_vki) > len(T_cki):
#     T_vki = T_vki[:len(T_cki)]
# elif len(T_vki) < len(T_cki):
#     T_cki = T_cki[:len(T_vki)]
# print(len(T_vki), len(T_cki))

scale = 1.0

#Set noise levels
trans_noise = 0.01 #Percentage of translation
rot_noise = 0.01 #Percentage of rotation

# Initialize solver
my_solver = solver()

# Convert inertial poses to relative
offset = 1
T_v_rel = inertial_to_relative(T_vki, offset=offset, reverse=True)
T_c_rel = inertial_to_relative(T_cki, offset=offset, reverse=True)
# T_v_rel = T_vki
# T_c_rel = T_cki

# Choose a subset of the data
num_poses_total = len(T_v_rel)
# num_poses = 3200
# subset_indices = np.random.choice(num_poses_total, num_poses, replace=False).tolist()
# T_v_rel_sub = add_noise([T_v_rel[index] for index in subset_indices], trans_noise, rot_noise)
# T_c_rel_sub = add_noise([T_c_rel[index] for index in subset_indices], trans_noise, rot_noise)
T_v_rel_sub = T_v_rel
T_c_rel_sub = T_c_rel
# T_v_rel_sub = [T_v_rel[index] for index in subset_indices]
# T_c_rel_sub = [T_c_rel[index] for index in subset_indices]

#Load the data into the solver
my_solver.set_T_v_rel(T_v_rel_sub) #Load relative egomotion sensor poses
my_solver.set_T_c_rel(T_c_rel_sub, scale=scale) # Load and scale relative camera sensor poses

# Run Solver
dual_time, dual_gt, dual_primal, dual_gap, dual_opt, dual_solution, dual_flag = my_solver.dual_solve(cons="RCH")
rel_time, rel_gt, rel_primal, rel_gap, relax_opt, relax_solution, rel_flag = my_solver.relax_solve(cons="RCH")

print("Scale: {}".format(scale))
# print(extcal)
print("Estimated Extrinsic Calibration:")
print(dual_solution, relax_solution)

if dual_flag:
    estimated_scale = 1/dual_opt[3]
    print("Estimated Scale: {}".format(1/dual_opt[3]))
    print("Dual solve: ", get_extrinsic_params_from_transformation_matrix(dual_solution, estimated_scale[0]))
if rel_flag:
    estimated_scale = 1/relax_opt[3]
    print("Estimated Scale: {}".format(1/relax_opt[3]))
    print("Relax solve: ", get_extrinsic_params_from_transformation_matrix(relax_solution, estimated_scale[0]))

