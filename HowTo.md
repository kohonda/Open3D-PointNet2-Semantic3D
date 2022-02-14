```
# Change cuda 10.0 in bashrc

# python 3.7
sudo update-alternatives --config python

source ~/venv/python3.7/pointnet/bin/activate

python visualize_features.py --ckpt weights/best_model_epoch_060.ckpt --kitti_root dataset/kitti_raw/

python output_features.py --kitti_root /home/honda/data/data_odometry_velodyne/dataset --sequence 03
```