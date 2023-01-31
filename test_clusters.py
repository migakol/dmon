import numpy as np
import os
import pickle

from signature_info import remove_bits_with_low_probability
from signature_info import compute_graph

def auto_clustering_pipeline():
    """
    Run clustering automatically
    """
    # Input definition and input loading
    root_data_folder = '/Users/michaelko/Code/dmon/data'
    signatures_file = os.path.join(root_data_folder, 'random_signatures.pkl')
    clusters_files = [os.path.join(root_data_folder, 'clusters_max_04.pkl')]

    # Get signatures
    with open(signatures_file, 'rb') as file:
        signatures = pickle.load(file)

    # Compute overall statistics
    # prior_probability = compute_prior_probability(signatures)

    # C
    # Step 1 - compute the prior probability of a bit to be on
    # all_bins = compute_bits_per_cluster(signatures, clusters, 0)
    # for cluster in range(1, 239):
    #     all_bins = all_bins + compute_bits_per_cluster(signatures, clusters, cluster)

def build_graph(signatures_file, base_save_name, low_prob_threshold=0.5, min_common_bits=80):
    root_data_folder = '/Users/michaelko/Code/dmon/data'
    signatures_file = os.path.join(root_data_folder, signatures_file)
    with open(signatures_file, 'rb') as file:
        signatures = pickle.load(file)

    # Compute prior probability and remove
    signatures = remove_bits_with_low_probability(signatures, low_prob_th=low_prob_threshold)
    pairs, distances = compute_graph(signatures, root_data_folder, base_save_name, min_common_bits=min_common_bits)

def unite_pairs_and_distances(distance_threshold, base_save_name, percentage=1.0):
    """
    Given pairs and distances saved by build graph function, unite them so that there is on pair file and one
    distance file
    The format is:
    f'dot_res_pairs_{k}.pkl', where k is from 10000 to 260000 with steps of 10K and
    'dot_product_graph_end.pkl'

    distance_threshold - we can save less distances than intended intially.
    percentage - how many points we want to use. If 1 - use all the points
    """
    root_data_folder = '/Users/michaelko/Code/dmon/data'
    numbers = np.linspace(5000, 50000, 10).astype('int')

    names = []
    for k in numbers:
        names.append(base_save_name + f'_{k}.pkl')
    names.append(base_save_name + '_end.pkl')

    num_signatures = 53959
    used_points = int(percentage * num_signatures)
    if used_points > num_signatures:
        used_points = num_signatures
    permutated_points = np.random.permutation(num_signatures)
    points_dict = {}
    for k in range(used_points):
        points_dict[permutated_points[k]] = k

    all_distances = []
    all_pairs = []
    for name in names:
        file = open(os.path.join(root_data_folder, name), 'rb')
        pairs, distances = pickle.load(file)
        file.close()

        if percentage < 1.0:
            new_pairs = []
            new_dist = []
            for k, pair in enumerate(pairs):
                if pair[0] in permutated_points[0:used_points] and pair[1] in permutated_points[0:used_points]:
                    new_pairs.append((points_dict[pair[0]], points_dict[pair[1]]))
                    # new_dist.append(distances[k].astype(np.float16))
                    new_dist.append(distances[k])

            indexes = (np.array(new_dist) > distance_threshold)
            new_dist = (np.array(new_dist)[indexes[0]]).tolist()
            new_pairs = (np.array(new_pairs)[indexes[0]]).tolist()

            all_pairs = all_pairs + new_pairs
            all_distances = all_distances + new_dist

        else:
            indexes = (np.array(distances) > distance_threshold)
            distances = (np.array(distances)[indexes[0]]).tolist()
            pairs = (np.array(pairs)[indexes[0]]).tolist()

            all_distances = all_distances + distances
            all_pairs = all_pairs + pairs

    # pickle.dump()
    dist_file = base_save_name + '_dist_file.pkl'
    pairs_file = base_save_name + '_pairs_file.pkl'
    file = open(os.path.join(root_data_folder, dist_file), 'wb')
    pickle.dump(all_distances, file)
    file.close()

    file = open(os.path.join(root_data_folder, pairs_file), 'wb')
    pickle.dump(all_pairs, file)
    file.close()

    print('Done')


def create_signatures_subset(main_signatures_file, percentage, output_signatures_sile):
    """
    Given the main signatures files, create a file with a smaller number of signatures to speeed up development
    """
    root_data_folder = '/Users/michaelko/Code/dmon/data'
    signatures_file = os.path.join(root_data_folder, main_signatures_file)
    with open(signatures_file, 'rb') as file:
        main_signatures = pickle.load(file)
    num_signatures = len(main_signatures)
    permutated_points = np.random.permutation(num_signatures)

    new_num = int(percentage * num_signatures)
    new_signatures = []
    for k in range(new_num):
        new_signatures.append(main_signatures[permutated_points[k]])

    with open(os.path.join(root_data_folder, output_signatures_sile), 'wb') as file:
        pickle.dump(new_signatures, file)


if __name__ == "__main__":
    print('Start clusterting')
    # create_signatures_subset('random_signatures.pkl', 0.2, 'random_signatures_20.pkl')
    # auto_clustering_pipeline()
    base_save_name = 'dot_prod_pair_dist_05_50'
    build_graph('random_signatures_20.pkl', base_save_name, low_prob_threshold=0.5, min_common_bits=50)
    # unite_pairs_and_distances(distance_threshold=60,base_save_name=base_save_name, percentage=1.0)
    print('Done clustering')