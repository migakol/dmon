import numpy as np
import pickle
import random
import copy
import time
from update_distance import update_distances, preprocess_data, distance_pt_to_cluster1, segmentation_internal_loop1

def preprocess_data1(pairs, num_points):
    """
    For every point, save all its pairs
    """
    point_hash = [[]] * num_points
    for k, pair in enumerate(pairs):
        point_hash[pair[0]].append(k)
        point_hash[pair[1]].append(k)

    return point_hash


def compute_distance_between_two_clusters(dist_array, pairs, point_hash, cluster1, cluster2, method):
    """
    method can be max, min, and average
    """
    pass
    # In this case, the distance between clusters is the minimal distance between two of their points
    if method == 'min':
        dist = 1000000
        for pt1 in cluster1:
            for pt2 in cluster2:
                if len(point_hash[pt1]) < len(point_hash[pt2]):
                    for pair_ind in point_hash[pt1]:
                        if pairs[pair_ind][0] == pt2 or pairs[pair_ind][1] == pt2:
                            dist = min(dist_array[pair_ind], dist)
                            break
                else:
                    for pair_ind in point_hash[pt2]:
                        if pairs[pair_ind][0] == pt1 or pairs[pair_ind][1] == pt1:
                            dist = min(dist_array[pair_ind], dist)
                            break
        if dist == 1000000:
            dist = 0
    elif method == 'max':
        dist = 0
        for pt1 in cluster1:
            for pt2 in cluster2:
                if len(point_hash[pt1]) < len(point_hash[pt2]):
                    for pair_ind in point_hash[pt1]:
                        if pairs[pair_ind][0] == pt2 or pairs[pair_ind][1] == pt2:
                            dist = max(dist_array[pair_ind], dist)
                            break
                else:
                    for pair_ind in point_hash[pt2]:
                        if pairs[pair_ind][0] == pt1 or pairs[pair_ind][1] == pt1:
                            dist = max(dist_array[pair_ind], dist)
                            break
    else:
        dist = 0
        dist_add = 0
        for pt1 in cluster1:
            for pt2 in cluster2:
                if len(point_hash[pt1]) < len(point_hash[pt2]):
                    for pair_ind in point_hash[pt1]:
                        if pairs[pair_ind][0] == pt2 or pairs[pair_ind][1] == pt2:
                            dist = dist + dist_array[pair_ind]
                            dist_add = dist_add + 1
                            break
                else:
                    for pair_ind in point_hash[pt2]:
                        if pairs[pair_ind][0] == pt1 or pairs[pair_ind][1] == pt1:
                            dist = dist + dist_array[pair_ind]
                            dist_add = dist_add + 1
                            break
        dist = dist / dist_add

    return dist

def fast_hierarchical_clustering(pairs, cluster_pairs, dist_array, method):
    cluster_list = []
    num_points = 269796
    # initial clusters
    clusters = []
    for k in range(num_points):
        clusters.append([k])
    cluster_list.append(clusters)

    # Preprocessing
    start = time.time()
    point_hash = preprocess_data(pairs, num_points)
    end = time.time()
    print('Preprocessing ', end - start)

    # Make lots of iterations
    for ind in range(num_points):
        # First pass:
        if ind == 0:
            cluster_dist_array = copy.deepcopy(dist_array)
            new_cluster_dist_array = [None] * (len(cluster_dist_array))
            new_cluster_pairs = [(None, None)] * (len(cluster_dist_array))
        else:
            if len(cluster_dist_array) < 50:
                break
        if ind > 1:
            break
            # cluster_pairs = copy.deepcopy(pairs)


        # from small to large - in our case large is close
        start = time.time()
        # arg_sort has the sorted index of cluster_dist_array from small to large - in our case large is close
        arg_sort = np.argsort(cluster_dist_array)
        end = time.time()
        print('Sorting ', end - start)
        # Merge the closest ones
        clusters = []
        # This is the map between old cluster number to new cluster numbers (after merge)
        old_2_new_cluster_map = np.zeros(len(cluster_list[-1]))
        new_ind = 0

        start = time.time()
        # These are the clusters with closest distance
        chosen_pair_cluster1 = cluster_pairs[arg_sort[-1]][0]
        chosen_pair_cluster2 = cluster_pairs[arg_sort[-1]][1]
        # cluster_list[-1] is the last cluster list - a list of lists
        # Each final list is a collection of signatures
        for k, cluster in enumerate(cluster_list[-1]):
            if k != chosen_pair_cluster1 and k != chosen_pair_cluster2:
                clusters.append(cluster)
                old_2_new_cluster_map[k] = new_ind
                new_ind = new_ind + 1
            else:
                old_2_new_cluster_map[k] = len(cluster_list[-1]) - 2
        merged_cluster = cluster_list[-1][cluster_pairs[arg_sort[-1]][0]] + cluster_list[-1][
            cluster_pairs[arg_sort[-1]][1]]
        end = time.time()
        print('Creating merged cluster ', end - start)

        start = time.time()
        new_cluster_dist_array, new_cluster_pairs, new_ind = update_distances(new_cluster_dist_array, new_cluster_pairs,
            cluster_dist_array, cluster_pairs, chosen_pair_cluster1, chosen_pair_cluster2, old_2_new_cluster_map)
        end = time.time()
        print('Update distances ', end - start)


def hierarchical_clustering(pairs, cluster_pairs, dist_array, method):
    cluster_list = []
    num_points = 269796
    # initial clusters
    clusters = []
    for k in range(num_points):
        clusters.append([k])
    cluster_list.append(clusters)

    # Preprocessing
    start = time.time()
    point_hash = preprocess_data(pairs, num_points)
    end = time.time()
    print('Preprocessing ', end - start)

    # Make lots of iterations
    for ind in range(num_points):
        # First pass:
        if ind == 0:
            cluster_dist_array = copy.deepcopy(dist_array)
        else:
            if len(cluster_dist_array) < 50:
                break
        if ind > 1:
            break
            # cluster_pairs = copy.deepcopy(pairs)
        # else:
        #     # Compute distance between all cluster pairs
        #     cluster_dist_array = []
        #     cluser_pairs = []
        #     for k in range(0, len(cluster_list[-1]) - 1):
        #         for m in range(k + 1, len(cluster_list[-1])):
        #             cluster_dist = compute_distance_between_two_clusters(dist_array, pairs, point_hash, cluster_list[-1][k],
        #                                                                  cluster_list[-1][m], method)
        #             cluster_dist_array.append(cluster_dist)
        #             cluster_pairs.append((k, m))

        # from small to large - in our case large is close
        start = time.time()
        # arg_sort has the sorted index of cluster_dist_array from small to large - in our case large is close
        arg_sort = np.argsort(cluster_dist_array)
        end = time.time()
        print('Sorting ', end - start)
        # Merge the closest ones
        clusters = []
        # This is the map between old cluster number to new cluster numbers (after merge)
        old_2_new_cluster_map = np.zeros(len(cluster_list[-1]))
        new_ind = 0

        start = time.time()
        # These are the clusters with closest distance
        chosen_pair_cluster1 = cluster_pairs[arg_sort[-1]][0]
        chosen_pair_cluster2 = cluster_pairs[arg_sort[-1]][1]
        # cluster_list[-1] is the last cluster list - a list of lists
        # Each final list is a collection of signatures
        for k, cluster in enumerate(cluster_list[-1]):
            if k != chosen_pair_cluster1 and k != chosen_pair_cluster2:
                clusters.append(cluster)
                old_2_new_cluster_map[k] = new_ind
                new_ind = new_ind + 1
            else:
                old_2_new_cluster_map[k] = len(cluster_list[-1]) - 2
        merged_cluster = cluster_list[-1][cluster_pairs[arg_sort[-1]][0]] + cluster_list[-1][
            cluster_pairs[arg_sort[-1]][1]]
        end = time.time()
        print('Creating merged cluster ', end - start)


        # Go over all distance
        start = time.time()
        new_cluster_dist_array = [None]*(len(cluster_dist_array))
        new_cluster_pairs = [(None, None)]*(len(cluster_dist_array))
        # This is the data that we can copy
        new_ind = 0
        for k in range(len(cluster_dist_array)):
            pair1 = cluster_pairs[k][0]
            pair2 = cluster_pairs[k][1]
            if pair1 != chosen_pair_cluster1 and pair1 != chosen_pair_cluster2 and \
                    pair2 != chosen_pair_cluster1 and pair2 != chosen_pair_cluster2:
                new_cluster_dist_array[new_ind] = cluster_dist_array[k]
                # new_cluster_dist_array.append(cluster_dist_array[k])
                new_cluster_pairs[new_ind] = (int(old_2_new_cluster_map[pair1]),
                                              int(old_2_new_cluster_map[pair2]))
                # new_cluster_pairs.append((int(old_2_new_cluster_map[cluster_pairs[k][0]]),
                #                          int(old_2_new_cluster_map[cluster_pairs[k][1]])))
                new_ind = new_ind + 1
        end = time.time()
        print('Updating distances ', end - start)

        start = time.time()
        new_cluster_dist_array = new_cluster_dist_array[:new_ind]
        new_cluster_pairs = new_cluster_pairs[:new_ind]
        end = time.time()
        print('Re-allocating ', end - start)

        # Merged cluster requires special treatment
        # Compute distance between merged cluster to all other clusters
        start = time.time()
        for k, kcluster in enumerate(clusters):
            cluster_dist = compute_distance_between_two_clusters(dist_array, pairs, point_hash, kcluster,
                                                                 merged_cluster, method)
            if cluster_dist >= 0.2:
                new_cluster_dist_array.append(cluster_dist)
                new_cluster_pairs.append((k, len(cluster_list[-1]) - 2))
        end = time.time()
        print('Computing new distances ', end - start)

        clusters.append(merged_cluster)
        cluster_list.append(clusters)
        cluster_dist_array = new_cluster_dist_array
        cluster_pairs = new_cluster_pairs

        # Delete to save memoryß
        if ind % 10 == 0 and ind > 0:
            print('Iteration ', ind)
            del cluster_list[-3]
            del cluster_list[-6]
            del cluster_list[-9]

        file = open('/Users/michaelko/Code/dmon/data/cluster_res.pkl', 'wb')
        pickle.dump(cluster_list, file)
        file.close()

def hierarchical_test():
    method = 'min'
    file = open('/Users/michaelko/Code/dmon/data/pairs_dist.pkl', 'rb')
    dist_array = pickle.load(file)
    file.close()
    file = open('/Users/michaelko/Code/dmon/data/res_pairs.pkl', 'rb')
    pairs = pickle.load(file)
    file.close()
    file = open('/Users/michaelko/Code/dmon/data/res_pairs.pkl', 'rb')
    pairs_tmp = pickle.load(file)
    file.close()

    fast_hierarchical_clustering(pairs, pairs_tmp, dist_array, 'min')


def initialize_clustering(point_hash):
    num_points = len(point_hash)
    random_points = np.random.permutation(num_points)
    # currently non is used
    used_points = np.zeros(num_points, dtype='int8')
    proc_points = np.zeros(num_points, dtype='int32')
    cur_random_cnt = -1
    iter = 0
    all_clusters = []

    return num_points, random_points, used_points, proc_points, cur_random_cnt, iter, all_clusters


def choose_free_pt(cur_random_cnt, num_points, used_points, random_points):
    if cur_random_cnt >= num_points - 1:
        return -1
    while True:
        if used_points[random_points[cur_random_cnt]] == 0:
            break
        else:
            cur_random_cnt = cur_random_cnt + 1

    return cur_random_cnt


def add_pt_to_stack(initial_stack, used_points, proc_points, cluster, pt_id, cur_cluster):
    initial_stack.append(pt_id)
    used_points[pt_id] = 1
    proc_points[pt_id] = cur_cluster
    cluster.append(pt_id)

    return initial_stack, used_points, proc_points, cluster


def fill_initial_stack(random_points, cur_random_cnt, used_points, point_hash, proc_points, pairs, cur_cluster,
                       dist_array, threshold):
    # At this stage, we have the initial point
    # Fill initial stack
    init_point = random_points[cur_random_cnt]
    used_points[init_point] = 1
    proc_points[init_point] = cur_cluster

    initial_stack = []
    cluster = []

    cluster.append(init_point)
    for pt in point_hash[init_point]:
        # Check both members of the pair
        proc_points[pairs[pt][0]] = cur_cluster
        proc_points[pairs[pt][1]] = cur_cluster
        if dist_array[pt] < threshold:
            continue
        if used_points[pairs[pt][0]] == 0:
            initial_stack, used_points, proc_points, cluster = add_pt_to_stack(initial_stack, used_points, proc_points,
                                                                               cluster, pairs[pt][0], cur_cluster)
        if used_points[pairs[pt][1]] == 0:
            initial_stack, used_points, proc_points, cluster = add_pt_to_stack(initial_stack, used_points, proc_points,
                                                                               cluster, pairs[pt][1], cur_cluster)

    return used_points, proc_points, initial_stack, cluster


def distance_pt_to_cluster(pt_id, cluster, dist_array, point_hash, pairs, method):
    # Create a list of the point's neighbors
    neighbors = []
    dist_dict = {}
    for pt in point_hash[pt_id]:
        if pairs[pt][0] == pt_id:
            neighbors.append(pairs[pt][1])
            dist_dict[pairs[pt][1]] = dist_array[pairs[pt][1]]
        else:
            neighbors.append(pairs[pt][0])
            dist_dict[pairs[pt][0]] = dist_array[pairs[pt][0]]

    # Go over the cluster
    inter = set(cluster).intersection(neighbors)
    distances = [dist_dict[x] for x in inter]

    if method == 'min':
        return min(distances)
    elif method == 'max':
        return max(distances)
    else:
        return sum(distances) / len(distances)


def segmentation_internal_loop(proc_points, used_points, initial_stack, point_hash, pairs, cur_cluster, dist_array,
                               cluster, method, threshold):
    while len(initial_stack) > 0:
        cur_proc_pt = initial_stack.pop()
        # Go over the neighborhood of cur_proc_pt
        for pt in point_hash[cur_proc_pt]:
            if proc_points[pairs[pt][0]] < cur_cluster:
                pt_to_test = pairs[pt][0]
            elif proc_points[pairs[pt][1]] < cur_cluster:
                pt_to_test = pairs[pt][1]
            else:
                continue
            if used_points[pt_to_test] != 0:
                continue
            # Compute the distance between pt_to_test to the cluster
            distance = distance_pt_to_cluster1(pt_to_test, cluster, dist_array, point_hash, pairs, method)

            if distance > threshold:
                initial_stack, used_points, proc_points, cluster = add_pt_to_stack(initial_stack, used_points,
                                                                                   proc_points, cluster, pt_to_test,
                                                                                   cur_cluster)

    return initial_stack, used_points, proc_points, cluster


def create_clusters(pairs, point_hash, dist_array, method, threshold):

    # initializations
    num_points, random_points, used_points, proc_points, cur_random_cnt, iter, all_clusters = initialize_clustering(point_hash)

    start = time.time()
    # main loop
    while iter < num_points:
    # while iter < 1:
    #     start_iter = time.time()
        if iter % 1000 == 0:
            end = time.time()
            print('Iteration ', iter, ' Clustering with ', threshold, ' takes ', end - start, ' Total clusters ',
                  len(all_clusters))
            print(iter)
        # Loop inits
        cur_random_cnt = cur_random_cnt + 1
        cur_cluster = len(all_clusters) + 1

        # Choose initial point
        cur_random_cnt = choose_free_pt(cur_random_cnt, num_points, used_points, random_points)
        if cur_random_cnt < 0:
            break

        # Fill inital stack
        used_points, proc_points, initial_stack, cluster = fill_initial_stack(random_points, cur_random_cnt,
                                    used_points, point_hash, proc_points, pairs, cur_cluster, dist_array, threshold)

        initial_stack, used_points, proc_points, cluster = segmentation_internal_loop1(proc_points, used_points,
                        initial_stack, point_hash, pairs, cur_cluster, dist_array, cluster, method, threshold)
        # while len(initial_stack) > 0:
        #     cur_proc_pt = initial_stack.pop()
        #     # Go over the neighborhood of cur_proc_pt
        #     for pt in point_hash[cur_proc_pt]:
        #         if proc_points[pairs[pt][0]] < cur_cluster:
        #             pt_to_test = pairs[pt][0]
        #         elif proc_points[pairs[pt][1]] < cur_cluster:
        #             pt_to_test = pairs[pt][1]
        #         else:
        #             continue
        #         if used_points[pt_to_test] != 0:
        #             continue
        #         # Compute the distance between pt_to_test to the cluster
        #         distance = distance_pt_to_cluster1(pt_to_test, cluster, dist_array, point_hash, pairs, method)
        #
        #         if distance > threshold:
        #             initial_stack, used_points, proc_points, cluster = add_pt_to_stack(initial_stack, used_points,
        #                                                             proc_points, cluster, pt_to_test, cur_cluster)

        iter = iter + 1
        if len(cluster) > 1:
            # end_iter = time.time()
            # iter_time = end_iter - start_iter
            # print('Single iter time ', iter_time, ' Cluster ', cluster[0], ' Cluster size ', len(cluster))
            all_clusters.append(cluster)

    end = time.time()
    print('Clustering with ', threshold, ' takes ', end - start)

    return all_clusters

def segmentation():
    file = open('/Users/michaelko/Code/dmon/data/pairs_dist.pkl', 'rb')
    dist_array = pickle.load(file)
    file.close()
    file = open('/Users/michaelko/Code/dmon/data/res_pairs.pkl', 'rb')
    pairs = pickle.load(file)
    file.close()

    threshold = 0.5
    threshold_str = '05'
    random.seed(10)
    num_points = 269796

    point_hash = preprocess_data(pairs, num_points)

    all_clusters = create_clusters(pairs, point_hash, dist_array, 'min', threshold)

    file = open('/Users/michaelko/Code/dmon/data/clusters_' + threshold_str + '.pkl', 'wb')
    pickle.dump(all_clusters, file)

if __name__ == "__main__":
    print('Clustering')
    # hierarchical_test()
    segmentation()