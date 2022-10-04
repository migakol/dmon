from absl import app
from absl import flags
import numpy as np

from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import tensorflow.compat.v2 as tf
import dmon
import gcn
import metrics
from utilities.readwrite import load_npz
from utilities.readwrite import convert_scipy_sparse_to_sparse_tensor
import utils
from layers.gcn import GCN
from layers.mincut import MincutPooling
from layers.modularity import ModularityPooling
from layers.bilinear import Bilinear

# visualization
import sys
sys.path.append('/opt/homebrew/Cellar/graph-tool/2.45_2/lib/python3.10/site-packages')
sys.path.append('/opt/homebrew/Cellar/gtk+/2.24.33/lib/gtk-2.0')
import graph_tool.all as gt

import matplotlib.pyplot as plt

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'DMON', 'Model to run')
flags.DEFINE_string('graph_path', None, 'Input graph path.')
flags.DEFINE_list('architecture', [64], 'Network architecture in the format `a,b,c,d`.')
flags.DEFINE_float('collapse_regularization', 1, 'Collapse regularization.', lower_bound=0)
flags.DEFINE_float( 'dropout_rate', 0, 'Dropout rate for GNN representations.', lower_bound=0, upper_bound=1)
flags.DEFINE_integer('n_clusters', 16, 'Number of clusters.', lower_bound=0)
flags.DEFINE_integer('n_epochs', 500, 'Number of epochs.', lower_bound=0)
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.', lower_bound=0)
flags.DEFINE_float('train_size', 0.2, 'Training data proportion.', lower_bound=0)


def build_dmon(feature_size, n_nodes):
    """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.
        Args:
            feature_size - dimension of the node feature
            n_nodes - the number of nodes
        Old Args:
            input_features: A dense [n, d] Keras input for the node features.
            input_graph: A sparse [n, n] Keras input for the normalized graph.
            input_adjacency: A sparse [n, n] Keras input for the graph adjacency.
            Returns: Built Keras DMoN model.
    """
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    output = input_features
    for n_channels in FLAGS.architecture:
        output = gcn.GCN(n_channels)([output, input_graph])
    pool, pool_assignment = dmon.DMoN(FLAGS.n_clusters, collapse_regularization=FLAGS.collapse_regularization,
                                      dropout_rate=FLAGS.dropout_rate)([output, input_adjacency])
    return tf.keras.Model(inputs=[input_features, input_graph, input_adjacency], outputs=[pool, pool_assignment])


def multilayer_gcn(inputs, channel_sizes):
    features, graph = inputs
    output = features
    for n_channels in channel_sizes:
        output = GCN(n_channels)([output, graph])
    return output


# TODO(tsitsulin): improve signature and documentation pylint: disable=dangerous-default-value,missing-function-docstring
def gcn_mincut(feature_size, n_nodes, channel_sizes, orthogonality_regularization=1, cluster_size_regularization=0, dropout_rate=0,
               pooling_mlp_sizes=[]):

    features = tf.keras.layers.Input(shape=(feature_size,))
    graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    output = features

    for n_channels in channel_sizes[:-1]:
        output = GCN(n_channels)([output, graph])
    pool, pool_assignment = MincutPooling( channel_sizes[-1], do_unpool=False,
                orthogonality_regularization=orthogonality_regularization,
                cluster_size_regularization=cluster_size_regularization, dropout_rate=dropout_rate,
                                           mlp_sizes=pooling_mlp_sizes)([output, graph])
    return tf.keras.Model(inputs=[features, graph], outputs=[pool, pool_assignment])


  # pylint: disable=missing-function-docstring
def deep_graph_infomax(feature_size, n_nodes):

    features = tf.keras.layers.Input(shape=(feature_size,))
    features_corrupted = tf.keras.layers.Input(shape=(feature_size,))
    graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    encoder = [GCN(64), GCN(32)]

    representations_clean = features
    representations_corrupted = features_corrupted

    for layer in encoder:
        representations_clean = layer([representations_clean, graph])
        representations_corrupted = layer([representations_corrupted, graph])

    representation_summary = tf.math.reduce_mean(representations_clean, axis=0)
    representation_summary = tf.nn.sigmoid(representation_summary)
    representation_summary = tf.reshape(representation_summary, [-1, 1])

    transform = Bilinear(representations_clean.shape[-1],
                       representations_clean.shape[-1])

    discriminator_clean = transform(
      [representations_clean, representation_summary])
    discriminator_corrupted = transform(
      [representations_corrupted, representation_summary])

    features_output = tf.concat([discriminator_clean, discriminator_corrupted], 0)

    return tf.keras.Model(inputs=[features, features_corrupted, graph],
                          outputs=[representations_clean, features_output])


# TODO(tsitsulin): improve signature and documentation pylint: disable=dangerous-default-value,missing-function-docstring
def gcn_modularity(feature_size, n_nodes, channel_sizes, orthogonality_regularization=0.3, cluster_size_regularization=1.0,
                   dropout_rate=0.75, pooling_mlp_sizes=[]):

    features = tf.keras.layers.Input(shape=(feature_size,))
    graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    output = features
    for n_channels in channel_sizes[:-1]:
        output = GCN(n_channels)([output, graph])
    pool, pool_assignment = ModularityPooling( channel_sizes[-1], do_unpool=False,
        orthogonality_regularization=orthogonality_regularization,
        cluster_size_regularization=cluster_size_regularization, dropout_rate=dropout_rate,
                                               mlp_sizes=pooling_mlp_sizes)([output, adjacency])
    return tf.keras.Model(inputs=[features, graph, adjacency], outputs=[pool, pool_assignment])


def compute_and_print_metrics(adjacency, clusters, labels, label_indices):
    conduct = metrics.conductance(adjacency, clusters)
    modular = metrics.modularity(adjacency, clusters)
    nmi = sklearn.metrics.normalized_mutual_info_score( labels, clusters[label_indices], average_method='arithmetic')
    precision = metrics.pairwise_precision(labels, clusters[label_indices])
    recall = metrics.pairwise_recall(labels, clusters[label_indices])
    f1 = 2 * precision * recall / (precision + recall)
    print('Conductance:', conduct)
    print('Modularity:', modular)
    print('NMI:', nmi)
    print('F1:', f1)

    return conduct, modular, nmi, f1


def compute_and_print_metrics_dgi(labels, label_indices, clusters, test_mask, adjacency):
    conduct = metrics.conductance(adjacency, clusters)
    modular = metrics.modularity(adjacency, clusters)
    nmi = sklearn.metrics.normalized_mutual_info_score(labels[test_mask[label_indices]], clusters,
                                                               average_method='arithmetic')
    precision = metrics.pairwise_precision(labels[test_mask[label_indices]], clusters)
    recall = metrics.pairwise_recall(labels[test_mask[label_indices]], clusters)
    F1 = 2 * precision * recall / (precision + recall)
    print('Conduct ', conduct)
    print('Modular ', modular)
    print('NMI:', nmi)
    print('F1 ', F1)


# Computes the gradients wrt. the sum of losses, returns a list of them.
def grad(model, inputs):
    with tf.GradientTape() as tape:
        _ = model(inputs, training=True)
        loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)


loss_object_dgi = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def loss_dgi(model, x, y, training):
    _, y_ = model(x, training=training)
    return loss_object_dgi(y_true=y, y_pred=y_)


def grad_dgi(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_dgi(model, inputs, targets, training=True)
        for loss_internal in model.losses:
            loss_value += loss_internal
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_test_train_masks(n_nodes, train_size):
    train_mask = np.zeros(n_nodes, dtype=np.bool)
    train_mask[np.random.choice(
        np.arange(n_nodes), int(n_nodes * train_size), replace=False)] = True
    test_mask = ~train_mask

    return train_mask, test_mask


def corrupt_data(features, epoch):
    data_corrupted = features.copy()
    perc_shuffle = np.linspace(1, 0.05, FLAGS.n_epochs)[epoch]
    # perc_shuffle = 1
    rows_shuffle = np.random.choice(np.arange(features.shape[1]), int(features.shape[1] * perc_shuffle))
    data_corrupted_tmp = data_corrupted[rows_shuffle]
    np.random.shuffle(data_corrupted_tmp)
    data_corrupted[rows_shuffle] = data_corrupted_tmp
    return data_corrupted

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Load and process the data (convert node features to dense, normalize the
    # graph, convert it to Tensorflow sparse tensor.
    adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)
    features = features.todense()
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(adjacency.copy()))

    if FLAGS.model_type == 'DMON':
        model = build_dmon(features.shape[1], adjacency.shape[0])
    elif FLAGS.model_type == 'MIN_CUT':
        model = gcn_mincut(features.shape[1], adjacency.shape[0], [64, 32, 16])
    elif FLAGS.model_type == 'MODULARITY':
        model = gcn_modularity(features.shape[1], adjacency.shape[0], [64, 32, 16])
    elif FLAGS.model_type == 'DGI':
        train_mask, test_mask = get_test_train_masks(features.shape[0], FLAGS.train_size)
        model = deep_graph_infomax(features.shape[1], adjacency.shape[0])
        labels_dgi = tf.concat([tf.zeros([features.shape[0], 1]), tf.ones([features.shape[0], 1])], 0)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
    model.compile(optimizer, None)

    epoch_arr = []
    loss_arr = []
    for epoch in range(FLAGS.n_epochs):
        if FLAGS.model_type == 'DMON':
            loss_values, grads = grad(model, [features, graph_normalized, graph])
        elif FLAGS.model_type == 'MIN_CUT':
            loss_values, grads = grad(model, [features, graph])
        elif FLAGS.model_type == 'MODULARITY':
            loss_values, grads = grad(model, [features, graph_normalized, graph])
        elif FLAGS.model_type == 'DGI':
            data_corrupted = corrupt_data(features, epoch)
            loss_values, grads = grad_dgi(model, [features, data_corrupted, graph], labels_dgi)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_arr.append(epoch)
        if FLAGS.model_type != 'DGI':
            print(f'epoch {epoch}, losses: ' + ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
            loss_arr.append(loss_values[1])
        else:
            print('epoch %d, loss: %0.4f' % (epoch, loss_values.numpy()))
            loss_arr.append(loss_values.numpy())

    # Obtain the cluster assignments.
    if FLAGS.model_type == 'DMON' or FLAGS.model_type == 'MODULARITY':
        _, assignments = model([features, graph_normalized, graph], training=False)
    if FLAGS.model_type == 'MIN_CUT':
        _, assignments = model([features, graph], training=False)
    elif FLAGS.model_type == 'DGI':
        representations, _ = model([features, data_corrupted, graph_normalized], training=False)


    if FLAGS.model_type != 'DGI':
        assignments = assignments.numpy()
        clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
        # Prints some metrics used in the paper.
        print('RESULTS ARE FOR: ', FLAGS.model_type)
        compute_and_print_metrics(adjacency, clusters, labels, label_indices)
    else:
        representations = representations.numpy()
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        clf.fit(representations[label_indices, :][train_mask[label_indices], :], labels[train_mask[label_indices]])
        clusters = clf.predict(representations[label_indices, :][test_mask[label_indices], :])
        compute_and_print_metrics_dgi(labels, label_indices, clusters, test_mask, adjacency)

def visualization_test():
    # Visualization part

    # First visualize the graph without clustering
    adjacency, features, labels, label_indices = load_npz('/Users/michaelko/Code/dmon/data/cora.npz')
    g = gt.Graph()
    num_nodes = features.shape[0]
    g.add_vertex(num_nodes)

    for row in range(num_nodes):
        cols = adjacency.indices[adjacency.indptr[row]:adjacency.indptr[row+1]]
        for col in cols:
            if col > row:
                g.add_edge(g.vertex(row), g.vertex(col))

    plot_color = g.new_vertex_property('vector<double>')
    g.vertex_properties['plot_color'] = plot_color

    cluster_map = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1), 2: (0, 1, 0, 1), 3: (0, 1, 1, 1), 4: (1, 0, 1, 1), 5: (1, 1, 0, 1),
                   6: (0.3, 0.5, 0.7, 1)}

    for ind, cluster in zip(label_indices, labels):
        plot_color[ind] = cluster_map[cluster]

    gt.graph_draw(g, vertex_text=g.vertex_index, vertex_fill_color=g.vertex_properties['plot_color'],
                  output="/Users/michaelko/Downloads/draw4.pdf")

    # Normalised RGB color.
    # 0->Red, 1->Blue
    # red_blue_map = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1)}
    # # Create new vertex property
    # plot_color = g.new_vertex_property('vector<double>')
    # # add that property to graph
    # g.vertex_properties['plot_color'] = plot_color
    # # assign a value to that property for each node of that graph
    # for v in g.vertices():
    #     plot_color[v] = red_blue_map[g.vertex_properties['value'][v]]
    #
    # gt.graph_draw(g,
    #               vertex_fill_color=g.vertex_properties['plot_color'])

    # import graph_tool
    # pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
    # pos = gt.sfdp_layout(g)
    # pos = gt.planar_layout(g)
    # pos = gt.radial_tree_layout(g, g.vertex(10))

    # g = gt.collection.data["netscience"]
    # g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    # state = gt.minimize_nested_blockmodel_dl(g)
    # t = gt.get_hierarchy_tree(state)[0]
    # tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1, use_index=False), weighted=True)
    # cts = gt.get_hierarchy_control_points(g, t, tpos)
    # pos = g.own_property(tpos)

    # state = gt.minimize_nested_blockmodel_dl(g)

    # pos = gt.arf_layout(g, max_iter=0)
    # gt.graph_draw(g, pos=pos, vertex_text=g.vertex_index, output="/Users/michaelko/Downloads/draw3.pdf")
    # graph_tool.draw.arf_layout(g, output="/Users/michaelko/Downloads/draw3.pdf")
    pass

if __name__ == '__main__':
    visualization_test()
    # app.run(main)