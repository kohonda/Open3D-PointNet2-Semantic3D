import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import open3d
import scipy.spatial as ss
import tensorflow as tf
from sklearn.decomposition import PCA

import model
from dataset.kitti_dataset import KittiDataset, KittiDatasetFeatures
from tf_ops.tf_interpolate import interpolate_label_with_color
from util.features_util import gen_dense_features


def interpolate_dense_labels(sparse_points, sparse_labels, dense_points, k=3):
    sparse_pcd = open3d.PointCloud()
    sparse_pcd.points = open3d.Vector3dVector(sparse_points)
    sparse_pcd_tree = open3d.KDTreeFlann(sparse_pcd)

    dense_labels = []
    for dense_point in dense_points:
        result_k, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
            dense_point, k
        )
        knn_sparse_labels = sparse_labels[sparse_indexes]
        dense_label = np.bincount(knn_sparse_labels).argmax()
        dense_labels.append(dense_label)
    return dense_labels


class PredictInterpolator:
    def __init__(self, checkpoint_path, num_classes, hyper_params):
        # Get ops from graph
        with tf.device("/gpu:0"):
            # Placeholders
            pl_sparse_points_centered_batched, _, _ = model.get_placeholders(
                hyper_params["num_point"], hyperparams=hyper_params
            )
            pl_is_training = tf.placeholder(tf.bool, shape=())

            # Prediction
            pred, end_points = model.get_model(
                pl_sparse_points_centered_batched,
                pl_is_training,
                num_classes,
                hyperparams=hyper_params,
            )
            sparse_labels_batched = tf.argmax(pred, axis=2)
            # (1, num_sparse_points) -> (num_sparse_points,)
            sparse_labels = tf.reshape(sparse_labels_batched, [-1])
            sparse_labels = tf.cast(sparse_labels, tf.int32)
            sparse_colors = tf.one_hot(sparse_labels, num_classes)

            # Saver
            saver = tf.train.Saver()

            # Graph for interpolating labels
            # Assuming batch_size == 1 for simplicity
            pl_sparse_points_batched = tf.placeholder(tf.float32, (None, None, 3))
            sparse_points = tf.reshape(pl_sparse_points_batched, [-1, 3])
            pl_dense_points = tf.placeholder(tf.float32, (None, 3))
            pl_knn = tf.placeholder(tf.int32, ())
            dense_labels, dense_colors = interpolate_label_with_color(
                sparse_points, sparse_labels, pl_dense_points, pl_knn
            )
            # TODO: 特徴量の補完

        self.ops = {
            "pl_sparse_points_centered_batched": pl_sparse_points_centered_batched,
            "pl_sparse_points_batched": pl_sparse_points_batched,
            "pl_dense_points": pl_dense_points,
            "pl_is_training": pl_is_training,
            "pl_knn": pl_knn,
            "dense_labels": dense_labels,
            "dense_colors": dense_colors,
            "end_points": end_points,
            "sparse_labels": sparse_labels,
            "sparse_colors": sparse_colors
        }

        # Restore checkpoint to session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored")

    def predict_and_interpolate(
        self,
        sparse_points_centered_batched,
        sparse_points_batched,
        dense_points,
        run_metadata=None,
        run_options=None,
    ):
        dense_labels_val, dense_colors_val, end_points = self.sess.run(
            [self.ops["dense_labels"], self.ops["dense_colors"], self.ops["end_points"]],
            feed_dict={
                self.ops[
                    "pl_sparse_points_centered_batched"
                ]: sparse_points_centered_batched,
                self.ops["pl_sparse_points_batched"]: sparse_points_batched,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: 3,
                self.ops["pl_is_training"]: False,
            },
        )


        return dense_labels_val, dense_colors_val, end_points
    
    def predict(
        self,
        sparse_points_centered_batched,
        sparse_points_batched,
        dense_points,
        run_metadata=None,
        run_options=None,
    ):
        sparse_labels_val, sparse_colors_val, end_points = self.sess.run(
            [self.ops["sparse_labels"], self.ops["sparse_colors"], self.ops["end_points"]],
            feed_dict={
                self.ops[
                    "pl_sparse_points_centered_batched"
                ]: sparse_points_centered_batched,
                self.ops["pl_sparse_points_batched"]: sparse_points_batched,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: 3,
                self.ops["pl_is_training"]: False,
            },
        )
        print("sparse colors: ", sparse_colors_val)
        print("sparse labels: ", sparse_labels_val)
        return sparse_labels_val, sparse_colors_val, end_points


def gen_random_color():
    color = np.random.rand(3)
    return color


def coloring_similar_feature_points(points, features, target_point_idx_list, coloring_points_num):
    #  coloring by feature points
    points_colors = np.zeros((len(points), 3))
    
    for target_point_idx in target_point_idx_list:
        tree = ss.KDTree(features)    
        _, index = tree.query(features[target_point_idx], coloring_points_num)
        points_colors[index] = gen_random_color()
   
    return points_colors


if __name__ == "__main__":
    np.random.seed(0)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="# samples, each contains num_point points",
    )
    parser.add_argument("--ckpt", default="weights/best_model_epoch_060.ckpt", help="Checkpoint file")
    parser.add_argument(
        "--kitti_root", default="", help="Checkpoint file", required=True
    )
    parser.add_argument("--sequence", default="00", help="sequence")
    parser.add_argument("--output_dir", default="/home/honda/data/kitti_features_pointnet", help="Output dir")
    
    
    flags = parser.parse_args()
    hyper_params = json.loads(open("semantic_no_color.json").read())

    # Create output dir
    if not os.path.exists(flags.output_dir):
        os.makedirs(flags.output_dir)

    output_folder = os.path.join(flags.output_dir, flags.sequence)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("Output dir: ", output_folder)

    # Dataset
    dataset = KittiDatasetFeatures(
        base_dir=flags.kitti_root,
        sequence_num=flags.sequence
    )

    # Model
    max_batch_size = 128  # The more the better, limited by memory size
    predictor = PredictInterpolator(
        checkpoint_path=flags.ckpt,
        num_classes=dataset.num_classes,
        hyper_params=hyper_params,
    )

    DATA_ID = 1
    print("DATA_ID: ", DATA_ID)
    kitti_file_data = dataset.list_file_data[DATA_ID]


    print("raw points size: ", kitti_file_data.points.shape)

    # Get data
    points_centered, points = kitti_file_data.get_batch_of_one_z_box_from_origin(
            num_points_per_sample=hyper_params["num_point"]
        )
    if len(points_centered) > max_batch_size:
        raise NotImplementedError("TODO: iterate batches if > max_batch_size")

    print("sampled points size: ", points.shape)


    # Predict and interpolate
    dense_points = kitti_file_data.points
    dense_labels, dense_colors, end_points = predictor.predict_and_interpolate(
            sparse_points_centered_batched=points_centered,  # (batch_size, num_sparse_points, 3)
            sparse_points_batched=points,  # (batch_size, num_sparse_points, 3)
            dense_points=dense_points,  # (num_dense_points, 3)
        )

    print("dense points size: ", dense_points.shape)
    
    # visualize dense points
    # pcd.points = open3d.Vector3dVector(dense_points)
    # pcd.colors = open3d.Vector3dVector(dense_colors.astype(np.float64))

    sparse_points = points_centered[0]

    # 特徴量を取得
    sparse_features = end_points["feats"][0]


    # 次元圧縮
    pca = PCA(n_components=4)
    compressed_features = pca.fit_transform(sparse_features)
    print("Raw features shape: ", sparse_features.shape)
    print("compressed features size: ", compressed_features.shape)
    # 寄与率
    print("explained variance ratio: ", pca.explained_variance_ratio_)
    print("accumulated variance ratio: ", pca.explained_variance_ratio_.sum())


    # Denseな特徴量を復元
    # TODO: 圧縮する前にinterpolate すること
    dense_features = gen_dense_features(dense_points, sparse_points, compressed_features)
    print("dense_features size: ", dense_features.shape)

    # 特徴量をファイルごとに出力
    input_file_path = dataset.file_list[DATA_ID]
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]  
    output_file = os.path.join(output_folder, "{}.txt".format(file_name))
    np.savetxt(output_file, dense_features, fmt="%.6f")

    # 特徴量を読み込み
    dense_compressed_features = np.loadtxt(output_file)
    print("Dense compressed features shape: ", dense_compressed_features.shape)

    def pick_points(pcd):
        print("")
        print(
            "1) Please pick point by [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = open3d.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()
    
    # visualize raw points
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(dense_points)
    points_colors = np.zeros((len(dense_points), 3)) # Black
    pcd.colors = open3d.Vector3dVector(dense_colors)

    picked_points_index =  pick_points(pcd)
    print("picked points: ", picked_points_index)

    # visualize colored by features
    nearest_points_num = 10
    points_colors = coloring_similar_feature_points(dense_points, dense_compressed_features, picked_points_index, nearest_points_num)
    
    pcd.colors = open3d.Vector3dVector(points_colors)
    open3d.draw_geometries([pcd])

   