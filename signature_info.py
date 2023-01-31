"""
The class provides interface to get informaton about signatures and signature statistics
"""
import numpy as np
import pickle
import os
import time

SIGNATURE_LOCATION = 3
Y_LOCATION = 1
X_LOCATION = 2
SIGNATURE_DIMENSION = 10000

from update_distance import common_bits_distance

def compute_graph(signatures, root_data_folder, base_save_name, min_common_bits=50):
    """
    The function computes the pairs and the distances
    """
    TOTAL_SIGS = len(signatures)
    list_of_sets = [set(sig[3].tolist()) for sig in signatures]

    pairs = []
    distances = []
    start_iter = 0
    start = time.time()
    for k in range(start_iter, TOTAL_SIGS - 1):
        # Below is the code for small thresholds of MINIMUM_COMMON_BITS
        if k % 5000 == 0 and k > start_iter:
            print(k)
            end = time.time()
            print(end - start)
            file = open(os.path.join(root_data_folder, base_save_name + f'_{k}.pkl'), 'wb')
            pickle.dump((pairs, distances), file)
            pairs = []
            distances = []
        for m in range(k + 1, TOTAL_SIGS):
            dist = common_bits_distance(list_of_sets[k], list_of_sets[m])
            dist = 400 if dist > 400 else int(dist)
            if dist >= min_common_bits:
                pairs.append((k, m))
                distances.append(dist)

    end = time.time()
    print(end - start)
    file = open(os.path.join(root_data_folder, base_save_name + f'_end.pkl'), 'wb')
    pickle.dump((pairs, distances), file)
    return pairs, distances

def compute_bits_per_cluster(signatures, clusters, cluster):
    """
    signatures - there are 7 field per signature. The bits themselves are the third field
    """
    all_bins = np.zeros(SIGNATURE_DIMENSION)
    for sig in clusters[cluster]:
        for v in signatures[sig][SIGNATURE_LOCATION]:
            all_bins[int(v)] = all_bins[int(v)] + 1

    return all_bins

def compute_prior_probability(signatures):
    """
    signatures - The structure is described in readme.txt in /data
    The fucntion computes for every bit the chance that it is ON in general data. This is done to detect bits that
    are on in a variety of signatures regardless of the content
    """
    bits_prior_prob = np.zeros(SIGNATURE_DIMENSION)
    for sig in signatures:
        for bit in sig[SIGNATURE_LOCATION]:
            bits_prior_prob[int(bit)] = bits_prior_prob[int(bit)] + 1

    return bits_prior_prob / len(signatures)

def remove_bits_with_low_probability(signatures, low_prob_th=0.2):
    prior_prob = compute_prior_probability(signatures)
    bad_bits = set(np.where(prior_prob > low_prob_th)[0])

    for sig in signatures:
        sig[3] = np.array(list(set(sig[3].tolist()) - bad_bits))

    return signatures
