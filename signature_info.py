"""
The class provides interface to get informaton about signatures and signature statistics
"""
import numpy as np

BIT_FILED = 3
SIGNATURE_DIMENSION = 10000

def compute_bits_per_cluster(signatures, clusters, cluster):
    """
    signatures - there are 7 field per signature. The bits themselves are the third field
    """
    all_bins = np.zeros(SIGNATURE_DIMENSION)
    for sig in clusters[cluster]:
        for v in signatures[sig][BIT_FILED]:
            all_bins[int(v)] = all_bins[int(v)] + 1

    return all_bins

def compute_full_statistics(signatures, clusters):
    """
    signatures - there are 7 field per signature. The bits themselves are the third field
    """
    all_bins = np.zeros(SIGNATURE_DIMENSION)
    for cluster in clusters:
        for sig in cluster:
            for v in signatures[sig][BIT_FILED]:
                all_bins[int(v)] = all_bins[int(v)] + 1

    return all_bins