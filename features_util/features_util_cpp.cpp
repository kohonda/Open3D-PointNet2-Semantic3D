#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <nanoflann.hpp>

namespace py = pybind11;

/**
 * @brief
 *
 * @param dense_points (n, 3)
 * @param sparse_points (n, 3)
 * @param sparse_features (n , D)
 * @param k
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd interpolate_dense_features(
    const Eigen::MatrixXd& dense_points, const Eigen::MatrixXd& sparse_points,
    const Eigen::MatrixXd& sparse_features, const int k = 1) {
    // Create a nanoflann KD-tree for the sparse points
    using kdtree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>;

    kdtree kd_tree(3, sparse_points, 10);

    Eigen::MatrixXd dense_features =
        Eigen::MatrixXd::Zero(dense_points.rows(), sparse_features.cols());

    for (int i = 0; i < dense_points.rows(); i++) {
        Eigen::Vector3d p = dense_points.row(i);
        // Eigen::MatrixXd dists = (sparse_points.rowwise() -
        // p).rowwise().norm(); Eigen::VectorXd sorted_dists =
        // dists.array().abs().matrix().colwise().minCoeff(); Eigen::VectorXi
        // sorted_indices = dists.array().abs().matrix().colwise().argmin();
        // Eigen::VectorXd sorted_features =
        // sparse_features.row(sorted_indices); Eigen::VectorXd sorted_weights =
        // sorted_dists.array().pow(-k).matrix(); dense_features.row(i) =
        // sorted_features.transpose() * sorted_weights;

        // find nearest neighbors by nanoflann
        size_t nn = k;
        size_t k_indices[nn];
        double k_distances[nn];
        nanoflann::KNNResultSet<double> resultSet(nn);
        resultSet.init(k_indices, k_distances);
        kd_tree.index->findNeighbors(resultSet, p.data(),
                                     nanoflann::SearchParams());

        // Averate the sparse features of the k nearest neighbors
        Eigen::VectorXd sum_features =
            Eigen::VectorXd::Zero(sparse_features.cols());
        for (int j = 0; j < nn; j++) {
            sum_features += sparse_features.row(k_indices[j]);
        }

        dense_features.row(i) = sum_features / nn;
    }
    return dense_features;
}

PYBIND11_MODULE(features_util_cpp, m) {
    m.def("interpolate_dense_features", &interpolate_dense_features);
}