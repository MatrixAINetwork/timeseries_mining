from ..utils import quality_measures


def cal_best_quality(dist_class_list, dataset_distribution, dist_distribution=None, base_entropy=None):
    """

    :param dist_class_list:
    :param dataset_distribution:
    :param dist_distribution:
    :param base_entropy:
    :return:
    """
    if dist_distribution is None:
        dist_distribution = {}
        for dist, class_ in dist_class_list:
            dist_distribution[class_] = dist_distribution.get(class_, 0) + 1

    A_distribution = {}
    B_distribution = {}
    B_count = 0
    A_count = 0
    n_samples = 0
    class_visited = {}
    for class_, nums in dataset_distribution.items():
        A_distribution[class_] = 0
        B_distribution[class_] = nums
        B_count += B_distribution[class_]
        n_samples += dataset_distribution[class_]
        class_visited[class_] = False

    if base_entropy is None:
        base_entropy = quality_measures.cal_entropy_distribution(dataset_distribution)

    gain_best = -1
    dist_last = -1
    for dist, class_ in dist_class_list:
        A_distribution[class_] += 1
        A_count += 1
        B_distribution[class_] -= 1
        B_count += 1

        if ~class_visited[class_]:
            unassign_count = dataset_distribution[class_] - dist_distribution[class_]
            B_count_class = B_distribution[class_] - unassign_count
            A_count_class = A_distribution[class_]
            if A_count_class > B_count_class:
                # assign the unassigned instance to A set
                A_distribution[class_] += unassign_count
                B_distribution[class_] -= unassign_count
                A_count += unassign_count
                B_count -= unassign_count
                class_visited[class_] = True

        if dist != dist_last:
            A_prob = A_count / n_samples
            B_prob = B_count / n_samples
            A_entropy = quality_measures.cal_entropy_distribution(A_distribution)
            B_entropy = quality_measures.cal_entropy_distribution(B_distribution)

            gain = base_entropy - A_prob * A_entropy - B_prob * B_entropy
            if gain > gain_best:
                gain_best = gain

        dist_last = dist

    return gain_best


