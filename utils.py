# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for graph processing."""
import numpy as np
import scipy.sparse
from scipy.sparse import base


def normalize_graph(graph,
                    normalized = True,
                    add_self_loops = True):
  """Normalized the graph's adjacency matrix in the scipy sparse matrix format.

  Args:
    graph: A scipy sparse adjacency matrix of the input graph.
    normalized: If True, uses the normalized Laplacian formulation. Otherwise,
      use the unnormalized Laplacian construction.
    add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
      the graph.

  Returns:
    A scipy sparse matrix containing the normalized version of the input graph.
  """
  if add_self_loops:
    graph = graph + scipy.sparse.identity(graph.shape[0])
  degree = np.squeeze(np.asarray(graph.sum(axis=1)))
  if normalized:
    with np.errstate(divide='ignore'):
      inverse_sqrt_degree = 1. / np.sqrt(degree)
    inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
    inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
    return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
  else:
    with np.errstate(divide='ignore'):
      inverse_degree = 1. / degree
    inverse_degree[inverse_degree == np.inf] = 0
    inverse_degree = scipy.sparse.diags(inverse_degree)
    return inverse_degree @ graph

def load_npz(
    filename
):
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
    adjacency = scipy.sparse.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    features = scipy.sparse.csr_matrix(
        (loader['feature_data'], loader['feature_indices'],
         loader['feature_indptr']),
        shape=loader['feature_shape'])

    label_indices = loader['label_indices']
    labels = loader['labels']
  assert adjacency.shape[0] == features.shape[
      0], 'Adjacency and feature size must be equal!'
  assert labels.shape[0] == label_indices.shape[
      0], 'Labels and label_indices size must be equal!'
  return adjacency, features, labels, label_indices

