import numpy as np
from .basic import ShapeletTransformBasic


def estimate_min_max_length(series_list, class_list):
    index = list(range(len(series_list)))
    k = 10
    num_shapelet = 10
    cls_STB = ShapeletTransformBasic(n_shapelets=num_shapelet,
                                     min_shapelet_length=3,
                                     max_shapelet_length=len(series_list[0]),
                                     length_increment=2,
                                     position_increment=10)

    shapelets_all = []
    for i in range(10):
        np.random.shuffle(index)
        sub_series_list = series_list[index[:k]]
        sub_class_list = class_list[index[:k]]
        cls_STB.fit(sub_series_list, sub_class_list)
        shapelets = cls_STB.train()
        print("%s round: " % i)
        print("one of shapelet length: %s" % len(shapelets[0].content))
        shapelets_all.extend(shapelets)

    min_id = 25
    max_id = 75
    shapelets_all = sorted(shapelets_all, key=lambda x: x.length)
    min_length = len(shapelets_all[min_id].content)
    max_length = len(shapelets_all[max_id].content)

    return min_length, max_length