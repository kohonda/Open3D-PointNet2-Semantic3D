#!/bin/bash

SCRIPT_DIR=$(
    cd $(dirname $0)
    pwd
)

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 00

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 01

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 02

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 03

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 04

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 05

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 06

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 07

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 08

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 09

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 10
