import numpy as np
import scipy
import tensorflow.compat.v2 as tf

def load_npz(filename):
    """Loads an attributed graph with sparse features from a specified Numpy file.
    Args:
    filename: A valid file name of a numpy file containing the input data.

    Returns:
        A tuple (graph, features, labels, label_indices) with the sparse adjacency
        matrix of a graph, sparse feature matrix, dense label array, and dense label
        index array (indices of nodes that have the labels in the label array).
    """
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        features = scipy.sparse.csr_matrix((loader['feature_data'], loader['feature_indices'],
                                            loader['feature_indptr']), shape=loader['feature_shape'])

        label_indices = loader['label_indices']
        labels = loader['labels']
        assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
        assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
        return adjacency, features, labels, label_indices


def convert_scipy_sparse_to_sparse_tensor(matrix):
    """Converts a sparse matrix and converts it to Tensorflow SparseTensor.
    Args:matrix: A scipy sparse matrix.
    Returns:
        A ternsorflow sparse matrix (rank-2 tensor).
    """
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


