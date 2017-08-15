import numpy as np
from .base import *
from ..utils import distance
from ..utils import quality_measures
from .shapelet_entity import *
from joblib import Parallel, delayed, cpu_count
from ..utils import pairwise
from time import time
from ..tools import output


class ShapeletTransformBasic(ShapeletTransform):
    def __init__(self,
                 min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1,
                 dist_func=distance.euclidean, dist_func_params=None,
                 n_jobs=10,
                 **kwargs):
        self._init_params(min_shapelet_length=min_shapelet_length,
                          max_shapelet_length=max_shapelet_length,
                          length_increment=length_increment,
                          position_increment=position_increment,
                          dist_func=dist_func,
                          dist_func_params=dist_func_params,
                          n_jobs=n_jobs,
                          **kwargs)

    def set_distfunc(self, dist_func, dist_func_params=None):
        self._set_distfunc(dist_func=dist_func,
                           dist_func_params=dist_func_params)

    def fit(self, series_list, class_list):
        self._fit(series_list=series_list,
                  class_list=class_list)

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

    def save_shapelet(self, filename):
        self._save_shapelet(filename=filename)

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