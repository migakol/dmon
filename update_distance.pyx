def update_distances(new_cluster_dist_array, new_cluster_pairs, cluster_dist_array, cluster_pairs,
                     chosen_pair_cluster1, chosen_pair_cluster2, old_2_new_cluster_map):

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

    return new_cluster_dist_array, new_cluster_pairs, new_ind

def preprocess_data(pairs, num_points):
    """
    For every point, save all its pairs
    """

    point_hash = [[] for k in range(num_points)]
    for k, pair in enumerate(pairs):
        point_hash[pair[0]].append(k)
        point_hash[pair[1]].append(k)

    return point_hash

def add_pt_to_stack1(initial_stack, used_points, proc_points, cluster, pt_id, cur_cluster):
    initial_stack.append(pt_id)
    used_points[pt_id] = 1
    proc_points[pt_id] = cur_cluster
    cluster.append(pt_id)

    return initial_stack, used_points, proc_points, cluster

def fill_initial_stack1(random_points, cur_random_cnt, used_points, point_hash, proc_points, pairs, cur_cluster,
                       dist_array, threshold, method):
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

        if used_points[pairs[pt][0]] == 0:
            distance = distance_pt_to_cluster1(pairs[pt][0], cluster, dist_array, point_hash, pairs, method)

            if distance > threshold:
                initial_stack, used_points, proc_points, cluster = add_pt_to_stack1(initial_stack, used_points,
                                                                               proc_points, cluster, pairs[pt][0],
                                                                               cur_cluster)

        if used_points[pairs[pt][1]] == 0:
            distance = distance_pt_to_cluster1(pairs[pt][1], cluster, dist_array, point_hash, pairs, method)

            if distance > threshold:
                initial_stack, used_points, proc_points, cluster = add_pt_to_stack1(initial_stack, used_points,
                                                                               proc_points, cluster, pairs[pt][1],
                                                                               cur_cluster)

        # if dist_array[pt] < threshold:
        #     continue
        # if used_points[pairs[pt][0]] == 0:
        #     initial_stack, used_points, proc_points, cluster = add_pt_to_stack1(initial_stack, used_points, proc_points,
        #                                                                        cluster, pairs[pt][0], cur_cluster)
        # if used_points[pairs[pt][1]] == 0:
        #     initial_stack, used_points, proc_points, cluster = add_pt_to_stack1(initial_stack, used_points, proc_points,
        #                                                                        cluster, pairs[pt][1], cur_cluster)

    return used_points, proc_points, initial_stack, cluster


def distance_pt_to_cluster1(pt_id, cluster, dist_array, point_hash, pairs, method):
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
        if len(inter) != len(cluster):
            return 0
        return min(distances)
    elif method == 'max':
        return max(distances)
    else:
        return sum(distances) / len(cluster)


def segmentation_internal_loop1(proc_points, used_points, initial_stack, point_hash, pairs, cur_cluster, dist_array,
                               cluster, method, threshold):
    while len(initial_stack) > 0:
        cur_proc_pt = initial_stack.pop()
        # Go over the neighborhood of cur_proc_pt
        for pt in point_hash[cur_proc_pt]:
            if proc_points[pairs[pt][0]] < cur_cluster:
                pt_to_test = pairs[pt][0]
                proc_points[pairs[pt][0]] = cur_cluster
            elif proc_points[pairs[pt][1]] < cur_cluster:
                pt_to_test = pairs[pt][1]
                proc_points[pairs[pt][1]] = cur_cluster
            else:
                continue
            if used_points[pt_to_test] != 0:
                continue
            # Compute the distance between pt_to_test to the cluster
            distance = distance_pt_to_cluster1(pt_to_test, cluster, dist_array, point_hash, pairs, method)

            if distance > threshold:
                initial_stack, used_points, proc_points, cluster = add_pt_to_stack1(initial_stack, used_points,
                                                                                   proc_points, cluster, pt_to_test,
                                                                                   cur_cluster)

    return initial_stack, used_points, proc_points, cluster