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

"""TODO(tsitsulin): add headers, tests, and improve style."""
import collections

import numpy as np
from absl import app
from absl import flags
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf
import sklearn.metrics
from models.gcn_mincut import gcn_mincut
from synthetic_data.overlapping_gaussians import circular_gaussians
from utilities.graph import construct_knn_graph
from utilities.graph import normalize_graph
from utilities.graph import scipy_to_tf
from utilities.metrics import conductance
from utilities.metrics import modularity
from utilities.metrics import precision
from utilities.metrics import recall
import metrics

from utils import load_npz
import utils
from utilities.metrics import accuracy_score

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'graph_path',
    None,
    'Input graph path.')
flags.DEFINE_integer(
    'n_nodes', 1000, 'Number of nodes for the synthetic graph.', lower_bound=0)
flags.DEFINE_integer(
    'n_clusters',
    10,
    'Number of clusters for the synthetic graph.',
    lower_bound=0)
flags.DEFINE_integer(
    'n_epochs', 200, 'Number of epochs to train.', lower_bound=0)
flags.DEFINE_float(
    'learning_rate', 0.01, 'Optimizer\'s learning rate.', lower_bound=0)

def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Bröther may i have some self-lööps')
  n_nodes = FLAGS.n_nodes
  n_clusters = FLAGS.n_clusters

  adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)

  features = features.todense()
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(
      utils.normalize_graph(adjacency.copy()))

  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = gcn_mincut([input_features, input_graph], [64, 32, 16])

  # data_clean, data_dirty, labels = circular_gaussians(n_nodes, n_clusters)
  # n_nodes = data_clean.shape[0]
  #
  # graph_clean_ = construct_knn_graph(data_clean)
  # graph_clean_normalized_ = normalize_graph(graph_clean_, normalized=True)
  #
  # graph_clean_normalized = scipy_to_tf(graph_clean_normalized_)
  #
  # input_features = tf.keras.layers.Input(shape=(2,))
  # input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  #
  # model = gcn_mincut([input_features, input_graph], [64, 32, 16])

  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  model.compile(optimizer, None)

  for epoch in range(FLAGS.n_epochs):
    # loss_value, grads = grad(model, [data_dirty, graph_clean_normalized])
    loss_value, grads = grad(model, [features, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, loss: {loss_value.numpy():.4f}')
  # _, assignments = model([data_dirty, graph_clean_normalized], training=False)
  _, assignments = model([features, graph], training=False)
  clusters = assignments.numpy().argmax(axis=1)

  print('Conductance:', metrics.conductance(adjacency, clusters))
  print('Modularity:', metrics.modularity(adjacency, clusters))

  # print('Conductance:', conductance(graph_clean_, clusters))
  # print('Modularity:', modularity(graph_clean_, clusters))

  print(
      'NMI:',
      sklearn.metrics.normalized_mutual_info_score(
          labels, clusters[label_indices], average_method='arithmetic'))
  precision = metrics.pairwise_precision(labels, clusters[label_indices])
  recall = metrics.pairwise_recall(labels, clusters[label_indices])
  # plt.plot(epoch_arr, loss_arr)
  print('F1:', 2 * precision * recall / (precision + recall))

  print(
      'NMI:',
      normalized_mutual_info_score(
          labels, clusters, average_method='arithmetic'))
  prec = precision(labels, clusters)
  rec = recall(labels, clusters)
  print('Precision:', prec)
  print('Recall:', rec)
  print('Cluster sizes for %d clusters:' % len(set(clusters)))
  print(collections.Counter(clusters))
  print('F1:', 2 * prec * rec / (prec + rec))


if __name__ == '__main__':
  app.run(main)
