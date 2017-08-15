import numpy as np
from ..utils import distance
from ..utils import quality_measures
from .shapelet_entity import *
from joblib import Parallel, delayed, cpu_count
from ..utils import pairwise
from time import time
from ..tools import output


def estimate_min_max_length(series_list, class_list):
    index = list(range(len(series_list)))
    k = 10
    cls_STS = ShapeletTransformSimplicity(min_shapelet_length=3,
                                          max_shapelet_length=len(series_list[0]),
                                          length_increment=2,
                                          position_increment=10)

    shapelets_all = []
    for i in range(10):
        np.random.shuffle(index)
        sub_series_list = series_list[index[:k]]
        sub_class_list = class_list[index[:k]]
        cls_STS.fit(sub_series_list, sub_class_list)
        shapelets = cls_STS.find_best_shapelets(10)
        print("%s round: " % i)
        print("one of shapelet length: %s" % len(shapelets[0].content))
        shapelets_all.extend(shapelets)

    min_id = 25
    max_id = 75
    shapelets_all = sorted(shapelets_all, key=lambda x: x.length)
    min_length = len(shapelets_all[min_id].content)
    max_length = len(shapelets_all[max_id].content)

    return min_length, max_length


class ShapeletTransformSimplicity:
    def __init__(self,
                 min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1,
                 dist_func=distance.euclidean, dist_func_params=None,
                 n_jobs=10,
                 **kwargs):

        # basic parameter
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.length_increment = length_increment
        self.position_increment = position_increment
        self.n_jobs = n_jobs

        # distance function
        self.dist_func = None
        self.dist_func_params = None
        self.set_distfunc(dist_func, dist_func_params)

        # data
        self.series_list = None
        self.class_list = None
        self.n_samples = None

        # return variable
        self.best_shapelets_ = None
        self.best_shapelets_content_ = None

    def set_distfunc(self, dist_func, dist_func_params=None):
        if callable(dist_func):
            self.dist_func = dist_func
            if dist_func_params is None:
                self.dist_func_params = {}
            else:
                self.dist_func_params = dist_func_params
        else:
            raise ValueError("distance function is not callable!!")

    def fit(self, series_list, class_list):
        assert len(series_list) == len(class_list), "series list and class list not aligning"

        self.series_list = series_list
        self.class_list = class_list
        self.n_samples = len(class_list)

    def find_best_shapelets(self, k):
        best_shapelets = []
        for series_id in range(self.n_samples):
            candidate_shapelets = []
            for win in range(self.min_shapelet_length, self.max_shapelet_length + 1, self.length_increment):
                candidate_shapelets.extend(self.generate_candidates(series_id, win))
            for shapelet in candidate_shapelets:
                distance_list = self.cal_subsequence_distance_batch(shapelet.content)
                quality = self.assess_candidate(distance_list)
                shapelet.quality = quality
                best_shapelets.append(shapelet)

            best_shapelets = sorted(best_shapelets, key=lambda x: x.quality, reverse=True)
            best_shapelets = self.remove_selfsimilar(best_shapelets)
            if len(best_shapelets) > k:
                best_shapelets = best_shapelets[:k]

        self._set_shapelets(best_shapelets)

        return best_shapelets

    def _set_shapelets(self, shapelets):
        self.best_shapelets_ = shapelets
        self.best_shapelets_content_ = []
        for i in range(len(shapelets)):
            self.best_shapelets_content_.append(shapelets[i].content)

    def save_shapelet(self, shapelets, filename):
        file = open(filename, 'w')
        for i in range(len(shapelets)):
            r_str = output.row2str(shapelets[i].content, ',')
            file.write(r_str + '\n')

    def transform(self, shapelet_list, series_list):
        params = {'position_increment': self.position_increment,
                  'dist_func': self.dist_func,
                  'dist_func_params': self.dist_func_params}
        transformed_ = []
        for shapelet in shapelet_list:
            if self.n_jobs == 1:
                distance_arr = np.array([distance.dist_subsequence(subsequence=shapelet.content,
                                                                   wholeseries=series,
                                                                   **params)
                                         for series in series_list])
            else:
                distance_arr = pairwise.parallel_pairwise(shapelet.content,
                                                          series_list,
                                                          n_jobs=self.n_jobs,
                                                          func=distance.dist_subsequence,
                                                          func_params=params)
            transformed_.append(distance_arr)

        transformed_ = np.array(transformed_)
        return transformed_.T

    def generate_candidates(self, series_id, win):
        series = self.series_list[series_id]
        length = len(series)
        pos = 0
        shapelet_list = []
        while pos <= length - win:
            shapelet = ShapeletEntity(win, pos, series_id,
                                      self.class_list[series_id],
                                      series[pos:(pos + win)])
            shapelet_list.append(shapelet)
            pos += self.position_increment

        return shapelet_list

    def assess_candidate(self, distance_list):
        entropy_base = quality_measures.calShanonEntropy(self.class_list)
        infoGain_max = -np.inf
        best_split_index = -1
        n = float(len(self.class_list))
        for i in range(len(distance_list)):
            dist_split = distance_list[i]
            subset_a_indexs = [distance_list <= dist_split]
            subset_b_indexs = [distance_list > dist_split]
            subset_a = self.class_list[subset_a_indexs]
            subset_b = self.class_list[subset_b_indexs]
            entropy_a = quality_measures.calShanonEntropy(subset_a)
            entropy_b = quality_measures.calShanonEntropy(subset_b)
            entropy_current = (float(len(subset_a)) / n)*entropy_a + (float(len(subset_b)) / n) * entropy_b
            infoGain_current = entropy_base - entropy_current
            if infoGain_current > infoGain_max:
                infoGain_max = infoGain_current
                best_split_index = i

        return infoGain_max, best_split_index

    def cal_subsequence_distance_batch(self, shapelet_content):
        params = {'position_increment': self.position_increment,
                  'dist_func': self.dist_func,
                  'dist_func_params': self.dist_func_params}
        if self.n_jobs == 1:
            distance_list = []
            for i in range(self.n_samples):
                dist = distance.dist_subsequence(subsequence=shapelet_content,
                                                 wholeseries=self.series_list[i],
                                                 **params)
                distance_list.append(dist)
        else:
            distance_list = pairwise.parallel_pairwise(shapelet_content,
                                                       self.series_list,
                                                       n_jobs=self.n_jobs,
                                                       func=distance.dist_subsequence,
                                                       func_params=params)
        return distance_list

    def is_selfsimilar(self, shapeletA, shapeletB):
        if shapeletA.series_id != shapeletB.series_id:
            return False
        # if self.class_list[shapeletA.series_id] != self.class_list[shapeletB.series_id]:
        #     return False
        if (shapeletA.start_pos <= shapeletB.start_pos) \
                and (shapeletB.start_pos < shapeletA.start_pos + shapeletA.length):
            # the start position of shapeletB is enclosed by shapeletA
            return True
        if (shapeletB.start_pos <= shapeletA.start_pos) \
                and (shapeletA.start_pos < shapeletB.start_pos + shapeletB.length):
            return True

        return False

    def remove_selfsimilar(self, shapelet_list):
        new_shapelet_list = []
        n = len(shapelet_list)
        self_similar = [False]*n
        for i in range(n):
            if self_similar[i]:
                continue
            new_shapelet_list.append(shapelet_list[i])
            for j in range(i+1, n):
                if ~self_similar[i] and self.is_selfsimilar(shapelet_list[i], shapelet_list[j]):
                    self_similar[j] = True

        return new_shapelet_list