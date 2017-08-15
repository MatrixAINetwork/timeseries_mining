import numpy as np


def calShanonEntropy(labels):
    """

    :param labels:
    :return:
    """

    uniClass = np.unique(labels)
    n = float(len(labels))
    entropy = 0.0
    for ic in uniClass:
        num = np.sum(labels == ic)
        prob = float(num) / n
        entropy += -prob*np.log2(prob)

    return entropy


def cal_entropy_distribution(class_distribution):
    """

    :param class_distribution:
    :return:
        entropy
    """
    if len(class_distribution) == 0:
        return 0
    n = 0
    nums = []
    for i, item in enumerate(class_distribution.items()):
        if item[1] > 0:
            nums.append(item[1])
            n += item[1]

    nums = np.array(nums)
    probs = nums / float(n)
    entropy = np.sum(-probs*np.log2(probs))

    return entropy
