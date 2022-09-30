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
from absl import app
from absl import flags
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
import tensorflow.compat.v2 as tf
import metrics
from layers.gcn import GCN
from models.dgi import deep_graph_infomax
from utilities.graph  import construct_knn_graph
from synthetic_data.overlapping_gaussians import circular_gaussians
from synthetic_data.overlapping_gaussians import line_gaussians

from utilities.readwrite import load_npz
from utilities.readwrite import convert_scipy_sparse_to_sparse_tensor
import utils

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'n_nodes', 1000, 'Number of nodes for the synthetic graph.', lower_bound=0)
flags.DEFINE_integer(
    'n_clusters',
    2,
    'Number of clusters for the synthetic graph.',
    lower_bound=0)
flags.DEFINE_float(
    'train_size', 0.2, 'Training data proportion.', lower_bound=0)
flags.DEFINE_integer(
    'n_epochs', 400, 'Number of epochs to train.', lower_bound=0)
flags.DEFINE_float(
    'learning_rate', 0.01, 'Optimizer\'s learning rate.', lower_bound=0)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Bröther may i have some self-lööps')
  n_nodes = FLAGS.n_nodes
  n_clusters = FLAGS.n_clusters
  train_size = FLAGS.train_size
  # michkol
  # data_clean, data_dirty, labels = overlapping_gaussians(n_nodes, n_clusters)
  data_clean, data_dirty, labels = line_gaussians(n_nodes, n_clusters)
  graph_clean = construct_knn_graph(data_clean).todense().A1.reshape(
      n_nodes, n_nodes)

  # adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)
  # features = features.todense()
  # graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  # graph_normalized = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(adjacency.copy()))


  train_mask = np.zeros(n_nodes, dtype=np.bool)
  train_mask[np.random.choice(
      np.arange(n_nodes), int(n_nodes * train_size), replace=False)] = True
  test_mask = ~train_mask
  print(f'Data shape: {data_clean.shape}, graph shape: {graph_clean.shape}')
  print(f'Train size: {train_mask.sum()}, test size: {test_mask.sum()}')

  input_features = tf.keras.layers.Input(shape=(2,))
  input_features_corrupted = tf.keras.layers.Input(shape=(2,))
  input_graph = tf.keras.layers.Input((n_nodes,))

  encoder = [GCN(64), GCN(32)]
  model = deep_graph_infomax(
      [input_features, input_features_corrupted, input_graph], encoder)

  def loss(model, x, y, training):
    _, y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets, training=True)
      for loss_internal in model.losses:
        loss_value += loss_internal
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  labels_dgi = tf.concat([tf.zeros([n_nodes, 1]), tf.ones([n_nodes, 1])], 0)
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

  for epoch in range(FLAGS.n_epochs):
    data_corrupted = data_dirty.copy()
    perc_shuffle = np.linspace(1, 0.05, FLAGS.n_epochs)[epoch]
    # perc_shuffle = 1
    rows_shuffle = np.random.choice(
        np.arange(n_nodes), int(n_nodes * perc_shuffle))
    data_corrupted_tmp = data_corrupted[rows_shuffle]
    np.random.shuffle(data_corrupted_tmp)
    data_corrupted[rows_shuffle] = data_corrupted_tmp
    loss_value, grads = grad(model, [data_dirty, data_corrupted, graph_clean],
                             labels_dgi)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print('epoch %d, loss: %0.4f, shuffle %0.2f%%' % (
        epoch, loss_value.numpy(), 100 * perc_shuffle))
  representations, _ = model([data_dirty, data_corrupted, graph_clean],
                             training=False)
  representations = representations.numpy()
  clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
  clf.fit(representations[train_mask], labels[train_mask])
  clusters = clf.predict(representations[test_mask])
  print(
      'NMI:',
      normalized_mutual_info_score(
          labels[test_mask], clusters, average_method='arithmetic'))
  print('Accuracy:', 100*accuracy_score(labels[test_mask], clusters))

def main1(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Bröther may i have some self-lööps')
  n_nodes = FLAGS.n_nodes
  n_clusters = FLAGS.n_clusters
  train_size = FLAGS.train_size
  # michkol
  # data_clean, data_dirty, labels = overlapping_gaussians(n_nodes, n_clusters)
  # graph_clean = construct_knn_graph(data_clean).todense().A1.reshape(
  #     n_nodes, n_nodes)

  adjacency, features, labels, label_indices = load_npz('/Users/michaelko/Code/dmon/data/cora.npz')
  # graph_clean = features
  features = features.todense()
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(adjacency.copy()))
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]

  train_mask = np.zeros(n_nodes, dtype=np.bool)
  train_mask[np.random.choice(
      np.arange(n_nodes), int(n_nodes * train_size), replace=False)] = True
  test_mask = ~train_mask
  # print(f'Data shape: {data_clean.shape}, graph shape: {graph_clean.shape}')
  # print(f'Train size: {train_mask.sum()}, test size: {test_mask.sum()}')

  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_features_corrupted = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)

  # input_features = tf.keras.layers.Input(shape=(2,))
  # input_features_corrupted = tf.keras.layers.Input(shape=(2,))
  # input_graph = tf.keras.layers.Input((n_nodes,))

  encoder = [GCN(64), GCN(32)]
  model = deep_graph_infomax([input_features, input_features_corrupted, input_graph], encoder)
  # model = deep_graph_infomax([features, features, graph], encoder)

  def loss(model, x, y, training):
    _, y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets, training=True)
      for loss_internal in model.losses:
        loss_value += loss_internal
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  labels_dgi = tf.concat([tf.zeros([n_nodes, 1]), tf.ones([n_nodes, 1])], 0)
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

  for epoch in range(FLAGS.n_epochs):
    # data_corrupted = data_dirty.copy()
    data_corrupted = features.copy()
    perc_shuffle = np.linspace(1, 0.05, FLAGS.n_epochs)[epoch]
    # perc_shuffle = 1
    rows_shuffle = np.random.choice(np.arange(n_nodes), int(n_nodes * perc_shuffle))
    data_corrupted_tmp = data_corrupted[rows_shuffle]
    np.random.shuffle(data_corrupted_tmp)
    data_corrupted[rows_shuffle] = data_corrupted_tmp
    # loss_value, grads = grad(model, [data_dirty, data_corrupted, graph_clean], labels_dgi)
    loss_value, grads = grad(model, [features, data_corrupted, graph], labels_dgi)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print('epoch %d, loss: %0.4f, shuffle %0.2f%%' % (
        epoch, loss_value.numpy(), 100 * perc_shuffle))

  representations, _ = model([features, data_corrupted, graph_normalized],
                             training=False)
  representations = representations.numpy()
  clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
  # full_labels = np.zeros((train_mask.shape[0],))
  # full_labels[label_indices] = labels
  # clf.fit(representations[train_mask], full_labels[train_mask])
  clf.fit(representations[label_indices, :][train_mask[label_indices], :], labels[train_mask[label_indices]])


  # clusters = clf.predict(representations[test_mask])
  clusters = clf.predict(representations[label_indices, :][test_mask[label_indices], :])

  # print( 'NMI:', normalized_mutual_info_score(full_labels[test_mask], clusters, average_method='arithmetic'))
  # print('Accuracy:', 100*accuracy_score(full_labels[test_mask], clusters))

  print('NMI:', normalized_mutual_info_score(labels[test_mask[label_indices]], clusters, average_method='arithmetic'))
  print('Accuracy:', 100 * accuracy_score(labels[test_mask[label_indices]], clusters))
  conduct = metrics.conductance(adjacency, clusters)
  modular = metrics.modularity(adjacency, clusters)
  print('Conduct ', conduct)
  print('Modular ', modular)
  precision = metrics.pairwise_precision(labels[test_mask[label_indices]], clusters)
  recall = metrics.pairwise_recall(labels[test_mask[label_indices]], clusters)
  f1 = 2 * precision * recall / (precision + recall)
  print('F1 ', f1)

if __name__ == '__main__':
  app.run(main1)
