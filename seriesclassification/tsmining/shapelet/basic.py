from .base import *
from ..utils import distance


class ShapeletTransformBasic(ShapeletTransform):
    def __init__(self,
                 n_shapelets=10,
                 min_shapelet_length=1, max_shapelet_length=1,
                 length_increment=1, position_increment=1,
                 dist_func=distance.euclidean, dist_func_params=None,
                 n_jobs=10,
                 **kwargs):
        self._init_params(n_shapelets=n_shapelets,
                          min_shapelet_length=min_shapelet_length,
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

