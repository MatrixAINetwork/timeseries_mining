import numpy as np
from ..utils import distance
from ..utils import quality_measures
from .shapelet_entity import ShapeletEntity
from .bounding import cal_best_quality


class ShapeletTransformSimplicity2:
    def __init__(self,
                 series_list,
                 class_list,
                 class_distribution=None,
                 min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1,
                 dist_func=distance.euclidean, dist_func_params=None,
                 **kwargs):

        assert len(series_list) == len(class_list), \
            "series list and class list not aligning"

        # basic parameter
        self.series_list = series_list
        self.class_list = class_list
        self.n_samples = len(series_list)
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.length_increment = length_increment
        self.position_increment = position_increment

        # distance function setting
        self.dist_func = dist_func
        self.dist_func_params = dist_func_params
        if dist_func_params is None:
            self.dist_func_params = {}

        # calculation class distribution of data set
        self.class_distribution = class_distribution
        if class_distribution is None:
            self.set_class_distribution()
        self.base_entropy = quality_measures.cal_entropy_distribution(self.class_distribution)

        # return variable
        self.best_shapelets_ = None

    def set_class_distribution(self):

        assert len(self.class_list) > 0, "class list should be larger than 0"

        self.class_distribution = {}
        for c in self.class_list:
            self.class_distribution[c] = self.class_distribution.get(c, 0) + 1

    """
        find best shapelet main process
    """

    def find_best_shapelets(self, k):
        """
        :param k: the number of best shapelet should be selected
        :return: 
            best k shapelets in list
        """
        best_shapelets = []
        for series_id in range(self.n_samples):
            candidate_shapelets = self.generate_candidates(series_id)
            candidate_shapelets = sorted(candidate_shapelets, key=lambda x: x.quality, reverse=True)
            print(len(candidate_shapelets))
            candidate_shapelets = self.remove_selfsimilar(candidate_shapelets)
            print(len(candidate_shapelets))
            best_shapelets = self.merge_k_shapelets(best_shapelets,
                                                    candidate_shapelets,
                                                    k,
                                                    self.shapelet_comparator_quality)
            print(len(best_shapelets))

        self.best_shapelets_ = best_shapelets
        return best_shapelets

    """
        generate shapelet candidates
    """

    def generate_candidates(self, series_id):
        shapelet_list_all = []
        series = self.series_list[series_id]
        length = len(series)
        for win in range(self.min_shapelet_length, self.max_shapelet_length + 1, self.length_increment):
            pos = 0
            shapelet_list = []
            while pos <= length - win:
                candidate = series[pos:(pos + length)]
                distance_list = list(map(lambda ts: self.cal_subdistance(candidate, ts), self.series_list))  # O(n)
                quality = self.assess_candidate(distance_list)
                shapelet = ShapeletEntity(length=win,
                                          start_pos=pos,
                                          series_id=series_id,
                                          class_label=self.class_list[series_id],
                                          content=series[pos:(pos + win)],
                                          quality=quality)
                shapelet_list.append(shapelet)
                pos += self.position_increment

            shapelet_list_all.extend(shapelet_list)

        return shapelet_list_all

    def check_candidate(self, candidate, best_quality, percentage):
        base_entropy = quality_measures.cal_entropy_distribution(self.class_distribution)
        dist_class_list = []
        for i in range(len(self.series_list)):
            if best_quality < np.Inf and len(dist_class_list) / self.n_samples > percentage:
                best_gain = cal_best_quality(dist_class_list, self.class_distribution, base_entropy=base_entropy)
                if best_gain < best_quality:
                    return True
                dist = self.cal_subdistance(candidate, self.series_list[i])
                dist_class_list.append((dist, self.class_list[i]))
                j = len(dist_class_list) - 2
                while j >= 0:
                    if dist_class_list[j][0] > dist:
                        break
                    else:
                        dist_class_list[j+1] = dist_class_list[j]

                dist_class_list[j+1] = (dist, self.class_list[i])

        distance_list = [item[0] for item in dist_class_list]
        quality = self.assess_candidate(distance_list)
        return quality

    def prune_candidate(self, best_qaulity, dist_class_list, percentage):
        if best_qaulity == np.Inf or len(dist_class_list) / self.n_samples <= percentage:
            return False
        else:
            return cal_best_quality()

    def merge_k_shapelets(self, best_shapelets, candidate_shapelets, k, comparator):
        best_id = 0
        cand_id = 0
        best_shapelets_sofar = []
        k_cur = 0
        while best_id < len(best_shapelets) and cand_id < len(candidate_shapelets):
            if len(best_shapelets_sofar) == k:
                break
            if comparator(best_shapelets[best_id], candidate_shapelets[cand_id]):
                best_shapelets_sofar.append(best_shapelets[best_id])
                best_id += 1
            else:
                best_shapelets_sofar.append(candidate_shapelets[cand_id])
                cand_id += 1
            k_cur += 1

        if k_cur < k:
            while best_id < len(best_shapelets) and k_cur < k:
                best_shapelets_sofar.append(best_shapelets[best_id])
                best_id += 1
                k_cur += 1

            while cand_id < len(candidate_shapelets) and k_cur < k:
                best_shapelets_sofar.append(candidate_shapelets[cand_id])
                cand_id += 1
                k_cur += 1

        return best_shapelets_sofar

    """
        shapelet comparator
    """

    def shapelet_comparator_quality(self, shapelet1, shapelet2):
        # prefer the shapelet with higher quality
        return shapelet1.quality - shapelet2.quality

    def shapelet_comparator_quality_longer(self, shapelet1, shapelet2):
        # prefer the shapelet higher quality and longer
        ans = shapelet1.quality - shapelet2.quality
        if ans == 0:
            return shapelet1.length - shapelet2.length
        else:
            return ans

    def shapelet_comparator_quality_shorter(self, shapelet1, shapelet2):
        ans = shapelet1.quality - shapelet2.quality
        if ans == 0:
            return -(shapelet1.length - shapelet2.length)
        else:
            return ans

    def assess_candidate(self, distance_list):
        dist_class_list = [(dist, label) for dist, label in zip(distance_list, self.class_list)]
        dist_class_list = sorted(dist_class_list, key=lambda x: x[0])
        A_distribution = {}
        B_distribution = {}
        n = 0
        for key, num in self.class_distribution.items():
            A_distribution[key] = 0
            B_distribution[key] = num
            n += num

        assert len(distance_list) == n, "the length distance list not accord the number of samples !!"

        base_entropy = quality_measures.cal_entropy_distribution(self.class_distribution)
        A_count = 0
        B_count = n
        dist_cur = -1
        dist_last = -1
        gain_max = -np.Inf
        for i in range(len(dist_class_list)):
            dist_cur = dist_class_list[i][0]
            class_cur = dist_class_list[i][1]
            A_distribution[class_cur] += 1
            B_distribution[class_cur] -= 1
            A_count += 1
            B_count -= 1

            if dist_cur != dist_last:
                A_prob = A_count / n
                B_prob = B_count / n
                A_entropy = quality_measures.cal_entropy_distribution(A_distribution)
                B_entropy = quality_measures.cal_entropy_distribution(B_distribution)
                gain_cur = base_entropy - A_prob*A_entropy - B_prob*B_entropy
            if gain_cur > gain_max:
                gain_max = gain_cur

            dist_last = dist_cur

        return gain_max

    def cal_subdistance(self, shapelet_content, timeseries):
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

    """
        transformation main didn't change
    """

    def transform(self, shapelet_list, series_list):
        n_samples = len(series_list)
        transformed_list = []
        for i in range(n_samples):
            series = series_list[i]
            distance_list = []
            for shapelet in shapelet_list:
                dist = self.cal_subdistance(shapelet.content, series)
                distance_list.append(dist)
            transformed_list.append(distance_list)
        return transformed_list

    def remove_selfsimilar(self, shapelet_list):
        new_shapelet_list = []
        n = len(shapelet_list)
        self_similar = [False] * n
        for i in range(n):
            if self_similar[i]:
                continue
            new_shapelet_list.append(shapelet_list[i])
            for j in range(i + 1, n):
                if ~self_similar[i] and self.is_selfsimilar(shapelet_list[i], shapelet_list[j]):
                    self_similar[j] = True

        return new_shapelet_list

    def is_selfsimilar(self, shapeletA, shapeletB):
        if shapeletA.series_id != shapeletB.series_id:
            return False
        # if there exist overlap ?
        if (shapeletA.start_pos <= shapeletB.start_pos) \
                and (shapeletB.start_pos < shapeletA.start_pos + shapeletA.length):
            return True
        if (shapeletB.start_pos <= shapeletA.start_pos) \
                and (shapeletA.start_pos < shapeletB.start_pos + shapeletB.length):
            return True

        return False

