import argparse
import json
import os
import time

import numpy as np
import open3d
import scipy.spatial as ss
import tensorflow as tf

import model
from dataset.kitti_dataset import KittiDataset
from tf_ops.tf_interpolate import interpolate_label_with_color


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

        self.ops = {
            "pl_sparse_points_centered_batched": pl_sparse_points_centered_batched,
            "pl_sparse_points_batched": pl_sparse_points_batched,
            "pl_dense_points": pl_dense_points,
            "pl_is_training": pl_is_training,
            "pl_knn": pl_knn,
            "dense_labels": dense_labels,
            "dense_colors": dense_colors,
            "end_points": end_points
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
        print("dense points size: ", len(dense_points))
        print("sparce points size: ", sparse_points_batched.shape)
        print("end points size: ", end_points["feats"].shape)
        return dense_labels_val, dense_colors_val, end_points


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
    parser.add_argument("--ckpt", default="", help="Checkpoint file")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--kitti_root", default="", help="Checkpoint file", required=True
    )
    flags = parser.parse_args()
    hyper_params = json.loads(open("semantic_no_color.json").read())

    # Create output dir
    sparse_output_dir = os.path.join("result", "sparse")
    dense_output_dir = os.path.join("result", "dense")
    os.makedirs(sparse_output_dir, exist_ok=True)
    os.makedirs(dense_output_dir, exist_ok=True)

    # Dataset
    dataset = KittiDataset(
        num_points_per_sample=hyper_params["num_point"],
        base_dir=flags.kitti_root,
        dates=["2011_09_26"],
        # drives=["0095", "0001"],
        drives=["0001"],
        box_size_x=hyper_params["box_size_x"],
        box_size_y=hyper_params["box_size_y"],
    )

    # Model
    max_batch_size = 128  # The more the better, limited by memory size
    predictor = PredictInterpolator(
        checkpoint_path=flags.ckpt,
        num_classes=dataset.num_classes,
        hyper_params=hyper_params,
    )

    # Init visualizer
    pcd = open3d.PointCloud()
    # vis = open3d.Visualizer()
    # vis.create_window()
    # vis.add_geometry(dense_pcd)
    # render_option = vis.get_render_option()
    # render_option.point_size = 0.05

    kitti_file_data = dataset.list_file_data[0]

    # Get data
    points_centered, points = kitti_file_data.get_batch_of_one_z_box_from_origin(
            num_points_per_sample=hyper_params["num_point"]
        )
    if len(points_centered) > max_batch_size:
        raise NotImplementedError("TODO: iterate batches if > max_batch_size")

    print("raw points size: ", points.shape)

    # Predict and interpolate
    dense_points = kitti_file_data.points
    dense_labels, dense_colors, end_points = predictor.predict_and_interpolate(
            sparse_points_centered_batched=points_centered,  # (batch_size, num_sparse_points, 3)
            sparse_points_batched=points,  # (batch_size, num_sparse_points, 3)
            dense_points=dense_points,  # (num_dense_points, 3)
        )
    
    # visualize
    # pcd.points = open3d.Vector3dVector(dense_points)
    # pcd.colors = open3d.Vector3dVector(dense_colors.astype(np.float64))

    # Coloring by feature points
    raw_points = points_centered[0]
    features = end_points["feats"][0]
    points_colors = np.zeros((len(raw_points), 3)) # RGB
    for i in range(len(raw_points)):
        points_colors[i][0] = features[i][0]
        points_colors[i][1] = 0
        points_colors[i][2] = 0
        # points_colors[i] = features[i][0]

    # visualize raw points
    # print("raw points size: ", raw_points.shape)
    pcd.points = open3d.Vector3dVector(raw_points)
    # print("color size: ", points_colors.shape)
    pcd.colors = open3d.Vector3dVector(points_colors)

    open3d.draw_geometries([pcd])

    