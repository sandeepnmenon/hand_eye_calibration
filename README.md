# pose_alignment_module
Code apapted from https://github.com/ethz-asl/hand_eye_calibration/blob/master/LICENSE to run as a python package without ROS or Catkin

# certifiable_calibration
Extrinsic calibration python code adapted from https://github.com/utiasSTARS/certifiable-calibration

# Running code

## Compute aligned poses
```
python compute_aligned_poses.py \
  --poses_B_H_csv_file tf_poses_timestamped.csv \
  --poses_W_E_csv_file camera_poses_timestamped.csv \
  --time_offset_output_csv_file time_offset.csv \
  --aligned_poses_output_numpy True
```

## Compute extrinsic calibration
Change the path to the aligned poses file in ```T_vki_filename``` and ```T_cki_filename```
```
python nonoverlapping_hand_eye.py 
```

## Visualize trajectories
python visualisations/visualize_trajectories.py --imu_trajectory_filename imu_trajectory_filename --camera_trajectory_filename camera_trajectory_filename

## Extrinsic calibration
