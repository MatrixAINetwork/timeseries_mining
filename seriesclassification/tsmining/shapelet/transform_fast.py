import numpy as np
from utils import distance
from utils import quality_measures
from .shapelet_entity import *


class ShapeletTransformFast:
    def __init__(self, series_list, class_list, min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1, dist_func=distance.dist_euclidean):
        assert len(series_list) == len(class_list), \
            "series list and class list not aligning"
        self.series_list = series_list
        self.class_list = class_list
        self.n_samples = len(series_list)
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.length_increment = length_increment
        self.position_increment = position_increment
        self.dist_func = dist_func

    def find_best_shapelets(self, k):
        best_shapelets = []
        for series_id in range(self.n_samples):
            for win in range(self.min_shapelet_length, self.max_shapelet_length + 1, self.length_increment):
                candidate_shapelets = self.generate_candidates(series_id, win)
                for shapelet in candidate_shapelets:
                    distance_list = self.cal_subsequence_distance_batch(shapelet.content)
                    quality = self.assess_candidate(distance_list)
                    shapelet.quality = quality
                    best_shapelets.append(shapelet)
            best_shapelets = sorted(best_shapelets, key=lambda x: x.quality, reverse=True)
            best_shapelets = self.remove_self_similar(best_shapelets)
            if len(best_shapelets) > k:
                best_shapelets = best_shapelets[:k]

        return best_shapelets

    def transform(self, shapelet_list, series_list):
        n_samples = len(series_list)
        transformed_list = []
        for i in range(n_samples):
            series = series_list[i]
            distance_list = []
            for shapelet in shapelet_list:
                dist = self.cal_subsequence_distance(shapelet.content, series)
                distance_list.append(dist)
            transformed_list.append(distance_list)
        return transformed_list

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

    def generate_candidates_batch(self, win):
        shapelet_list_all = []
        for i in range(self.n_samples):
            shapelet_list_all.extend(self.generate_candidates(i, win))
        return shapelet_list_all

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

    def cal_subsequence_distance(self, shapelet_content, timeseries):
        dist_min = np.inf
        spLen = len(shapelet_content)
        tsLen = len(timeseries)
        pos = 0
        while pos <= tsLen - spLen:
            dist_current = self.dist_func(shapelet_content, timeseries[pos:(pos + spLen)], cut_value=dist_min)
            if dist_current < dist_min:
                dist_min = dist_current
            pos += self.position_increment
        return dist_min

    def cal_subsequence_distance_batch(self, shapelet_content):
        distance_list = []
        for i in range(self.n_samples):
            dist = self.cal_subsequence_distance(shapelet_content, self.series_list[i])
            distance_list.append(dist)
        return distance_list

    def is_self_similar(self, shapeletA, shapeletB):
        if self.class_list[shapeletA.series_id] != self.class_list[shapeletB.series_id]:
             return False
        if (shapeletA.start_pos <= shapeletB.start_pos) \
                and (shapeletB.start_pos < shapeletA.start_pos + shapeletA.length):
            # the start position of shapeletB is enclosed by shapeletA
            return True
        if (shapeletB.start_pos <= shapeletA.start_pos) \
                and (shapeletA.start_pos < shapeletB.start_pos + shapeletB.length):
            return True

        return False

    def remove_self_similar(self, shapelet_list):
        new_shapelet_list = []
        n = len(shapelet_list)
        self_similar = [False]*n
        for i in range(n):
            if self_similar[i]:
                continue
            new_shapelet_list.append(shapelet_list[i])
            for j in range(i+1, n):
                if ~self_similar[i] and self.is_self_similar(shapelet_list[i], shapelet_list[j]):
                    self_similar[j] = True

        return new_shapelet_list

