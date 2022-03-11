# hand_eye_calibration
Code apapted from https://github.com/ethz-asl/hand_eye_calibration/blob/master/LICENSE to run as a python package without ROS or Catkin

# Running code

## Compute aligned poses
```
python compute_aligned_poses.py \
  --poses_B_H_csv_file tf_poses_timestamped.csv \
  --poses_W_E_csv_file camera_poses_timestamped.csv \
  --time_offset_output_csv_file time_offset.csv \
  --aligned_poses_output_numpy True
```

## Visualize trajectories
python visualisations/visualize_trajectories.py --imu_trajectory_filename imu_trajectory_filename --camera_trajectory_filename camera_trajectory_filename