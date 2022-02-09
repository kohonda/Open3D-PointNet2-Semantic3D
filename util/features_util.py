from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.neighbors import NearestNeighbors



def calc_ave_feature(dense_point, sparse_points, sparse_features, k=1):
    # dense_point: (1, 3) float32 array, dense point
    # sparse_points: (num_sparse_points, 3) float32 array, sparse points
    # sparse_features: (num_sparse_points, N) float32 array, features of sparse_points
    # k: int, use k-NN for label interpolation
    # Output:dense_features: (1, N) float32 array, dense features
    dense_features = np.zeros((1, sparse_features.shape[1]))

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(sparse_points)
    distances, indices = nbrs.kneighbors([dense_point])
    dense_features[0] = np.mean(sparse_features[indices[0]], axis=0)

    return dense_features

def gen_dense_features(dense_points, sparse_points, sparse_features, k=1):
    # dense_points: (num_dense_points, 3) float32 array, dense points
    # sparse_points: (num_sparse_points, 3) float32 array, sparse points
    # sparse_features: (num_sparse_points, N) float32 array, features of sparse_points
    # k: int, use k-NN for label interpolation
    # Output:dense_features: (num_dense_points, N) float32 array, dense features
    dense_features = np.zeros((dense_points.shape[0], sparse_features.shape[1]))

    for i in range(dense_points.shape[0]):
        dense_point = dense_points[i]
        dense_features[i] = calc_ave_feature(dense_point, sparse_points, sparse_features, k)

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(calc_ave_feature, dense_points[i], sparse_points, sparse_features, k) for i in range(dense_points.shape[0])]
    #     for i, future in enumerate(futures):
    #         dense_features[i] = future.result()
        
    return dense_features

