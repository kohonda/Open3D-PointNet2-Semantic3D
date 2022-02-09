import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(__file__)
BUILD_DIR = os.path.join(BASE_DIR, "build")
sys.path.append(BUILD_DIR)
import features_util_cpp 


def interpolate_dense_features(dense_point, sparse_points, sparse_features, k=1):
    # dense_points: (num_dense_points, 3) float32 array, dense points
    # sparse_points: (num_sparse_points, 3) float32 array, sparse points
    # sparse_features: (num_sparse_points, N) float32 array, features of sparse_points
    # k: int, use k-NN for label interpolation
    # Output:dense_features: (num_dense_points, N) float32 array, dense features
    return features_util_cpp.interpolate_dense_features(dense_point, sparse_points, sparse_features, k)
