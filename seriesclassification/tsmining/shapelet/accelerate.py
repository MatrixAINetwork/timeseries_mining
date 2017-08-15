from .bounding import cal_best_quality
from .base import *


class ShapeletTransformPruning(ShapeletTransform):
    def __init__(self,
                 n_shapelets=10,
                 min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1,
                 dist_func=distance.euclidean, dist_func_params=None,
                 n_jobs=10,
                 **kwargs):
        """

        :param n_shapelets:
        :param min_shapelet_length:
        :param max_shapelet_length:
        :param length_increment:
        :param position_increment:
        :param dist_func:
        :param dist_func_params:
        :param n_jobs:
        :param kwargs:
        """
        self._init_params(n_shapelets=n_shapelets,
                          min_shapelet_length=min_shapelet_length,
                          max_shapelet_length=max_shapelet_length,
                          length_increment=length_increment,
                          position_increment=position_increment,
                          dist_func=dist_func,
                          dist_func_params=dist_func_params,
                          n_jobs=n_jobs,
                          **kwargs)

        # additional parameter
        self.class_distribution = None
        self.base_entropy = None

    def set_class_distribution(self):
        """
            calculate class distribution for class list
        :return:
            dictionary {'class':number of record}
        """

        assert self.class_list is not None, "class list should be set!!"
        assert len(self.class_list) > 0, "class list should be larger than 0"

        self.class_distribution = {}
        for c in self.class_list:
            self.class_distribution[c] = self.class_distribution.get(c, 0) + 1

    def fit(self, series_list, class_list):
        """

        :param series_list:
        :param class_list:
        :return:
        """
        self._fit(series_list=series_list,
                  class_list=class_list)
        self.set_class_distribution()
        self.base_entropy = quality_measures.cal_entropy_distribution(self.class_distribution)

    def train(self):
        """

        :return:
        """
        k = self.n_shapelets
        best_shapelets = []
        pruning_quality = np.inf
        for series_id in range(self.n_samples):
            #candidate_shapelets = self.generate_candidates(series_id)
            candidate_shapelets = self.generate_candidates(series_id, pruning_quality, percentage=0.8)
            candidate_shapelets = sorted(candidate_shapelets, key=lambda x: x.quality, reverse=True)
            candidate_shapelets = self.remove_selfsimilar(candidate_shapelets)
            print("candidate shapelets length: ", len(candidate_shapelets))
            best_shapelets = self.merge_k_shapelets(best_shapelets,
                                                    candidate_shapelets,
                                                    k,
                                                    ShapeletComparator.cmp_quality)
            pruning_quality = best_shapelets[len(best_shapelets)-1].quality
            print("pruning quality: ", pruning_quality)
        self._set_shapelets(best_shapelets)

        return best_shapelets

    def generate_candidates(self, series_id, best_quality, percentage = 0.7):
        """

        :param series_id:
        :return:
        """
        shapelet_list_all = []
        series = self.series_list[series_id]
        length = len(series)
        for win in range(self.min_shapelet_length, self.max_shapelet_length + 1, self.length_increment):
            pos = 0
            shapelet_list = []
            while pos <= length - win:
                candidate = series[pos:(pos + length)]
                quality = self.check_candidate(candidate, best_quality, percentage)
                if quality > 0:
                     shapelet = ShapeletEntity(length=win,
                                               start_pos=pos,
                                               series_id=series_id,
                                               class_label=self.class_list[series_id],
                                               content=series[pos:(pos + win)],
                                               quality=quality)
                     shapelet_list.append(shapelet)
                # distance_list = list(map(lambda ts: self.cal_subdistance(candidate, ts), self.series_list))
                # quality = self.assess_candidate(distance_list)
                # shapelet = ShapeletEntity(length=win,
                #                           start_pos=pos,
                #                           series_id=series_id,
                #                           class_label=self.class_list[series_id],
                #                           content=series[pos:(pos + win)],
                #                           quality=quality)
                # shapelet_list.append(shapelet)
                pos += self.position_increment

            shapelet_list_all.extend(shapelet_list)

        return shapelet_list_all

    def check_candidate(self, candidate, best_quality, percentage=0.7):
        """

        :param candidate:
        :param best_quality:
        :param percentage: the threshold to control time to calculate the best information gain
                           when the number of distance candidate have been more than specific percentage of total
        :return:
        """
        #base_entropy = quality_measures.cal_entropy_distribution(self.class_distribution)
        base_entropy = self.base_entropy

        # list element : (distance between candidate and i-th time series, class label)
        # note dist_class_list is sorted by distance
        dist_class_list = []

        # traverse series_list to calculate the distance between each time series and candidate
        for i in range(len(self.series_list)):
            if best_quality < np.Inf and len(dist_class_list) / self.n_samples > percentage:
                # best quality has been set
                # and the required amount of data has been observed
                best_gain = cal_best_quality(dist_class_list, self.class_distribution, base_entropy=base_entropy)
                if best_gain < best_quality:  # pruning
                    return -1

            dist = self.cal_subdistance(candidate, self.series_list[i])
            dist_class_list.append((dist, self.class_list[i]))
            # insert new element into right place and insure list still in ascending order
            j = len(dist_class_list) - 2
            for j in range(len(dist_class_list)-2, -1, -1):
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

