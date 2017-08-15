"""
    class for k neighbor classification.
    author: FanLing Huang
    version: 1.0

"""
import numpy as np
from ..utils import distance
from ..utils import votingMethod
from ..utils import pairwise


class KNeighborsClassifier:
    """
    
    """
    def __init__(self, n_neighbors=5,
                 distfunc=distance.euclidean, distfunc_params=None,
                 votefunc=votingMethod.vote_majority, votefunc_params=None,
                 n_jobs=1, **kwargs):
        """
        :param n_neighbors: 
        :param distfunc: 
        :param distfunc_params: 
        :param votefunc: 
        :param votefunc_params: 
        :param n_jobs: 
        :param kwargs: 
        """

        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self._fit_X = None
        self._fit_y = None

        self.distfunc = None
        self.distfunc_params = None
        self.set_distfunc(distfunc, distfunc_params)

        self.votefunc = None
        self.votefunc_params = None
        self.set_votefunc(votefunc, votefunc_params)

        self.effective_distfunc_params_ = None
        self.effective_votefunc_params_ = None

    def set_distfunc(self, distfunc, distfunc_params=None):
        if callable(distfunc):
            self.distfunc = distfunc
        else:
            raise ValueError("distance function is not callable!")
        self.distfunc_params = distfunc_params
        if self.distfunc_params is None:
            self.effective_distfunc_params_ = {}
        else:
            self.effective_distfunc_params_ = self.distfunc_params.copy()

    def set_votefunc(self, votefunc, votefunc_params=None):
        if callable(votefunc):
            self.votefunc = votefunc
        else:
            raise ValueError("voting function is not callable!")
        self.votefunc_params = votefunc_params
        if self.votefunc_params is None:
            self.effective_votefunc_params_ = {}
        else:
            self.effective_votefunc_params_ = self.votefunc_params.copy()

    def fit(self, X, y):
        """
        
        :param X: 
        :param y: 
        :return: 
        """
        if self.distfunc_params is None:
            self.effective_distfunc_params_ = {}
        else:
            self.effective_distfunc_params_ = self.distfunc_params.copy()

        if self.votefunc_params is None:
            self.effective_votefunc_params_ = {}
        else:
            self.effective_votefunc_params_ = self.votefunc_params.copy()

        assert len(X) == len(y), "the length of train set doesn't math the length of class label"
        self._fit_X = X
        self._fit_y = y

        return self

    def predict(self, X):
        """
        
        :param X: 
        :return: 
        """
        assert self.n_neighbors <= len(self._fit_X), \
            "the number of neighbor larger than the number of training instance!"
        pred_list = []
        for ind, iX in enumerate(X):
            dist_list = pairwise.parallel_pairwise(iX, self._fit_X,
                                                   n_jobs=self.n_jobs,
                                                   func=self.distfunc,
                                                   func_params=self.effective_distfunc_params_)

            sorted_indexs = np.argsort(dist_list)
            nearest_classes = self._fit_y[sorted_indexs[:self.n_neighbors]]
            vote_class = self.votefunc(nearest_classes)
            pred_list.append(vote_class)

        return pred_list

