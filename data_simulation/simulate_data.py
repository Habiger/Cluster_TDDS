import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Cluster:
    #TODO add gaussian noise to cluster values
    #TODO add additional "noise measurements" (maybe?)
    #TODO norm distribution should start at 
    size_range = (1, 100)  # 1....1000 given from Prof Waltl
    scale_range_exp = (10**-3, 1)
    scale_range_norm = (0.1, 1)
    loc_range_norm = (0.5, 7)  # 


    def __init__(self):
        self.size = self._get_size()
        scale_exp = np.random.uniform(low=self.scale_range_exp[0], high=self.scale_range_exp[1])
        self.param_exp = {"scale": scale_exp, "size": self.size}

        self.scale_norm = np.random.uniform(low=self.scale_range_norm[0], high=self.scale_range_norm[1])
        self.loc_norm = np.random.uniform(low=self.loc_range_norm[0], high=self.loc_range_norm[1])

        self.param_norm = {"loc": self.loc_norm, "scale": self.scale_norm, "size": self.size}
        self.coordinates = self._get_coordinates()


    def _get_size(self):
        possible_size_numbers = [i for i in range(self.size_range[0], self.size_range[1]+1)]
        return random.choice(possible_size_numbers)

    def _get_coordinates(self):
        x = np.random.exponential(**self.param_exp)
        y = np.random.normal(**self.param_norm)
        return pd.DataFrame({"x": x, "y": y})



class Experiment:
    #TODO implement step sizes to locate normal distribution
      # no more than 10 Clusters useful according to DA


    def __init__(self, max_cluster_number=5):
        #np.random.seed(seed) doenst work yet
        self.n_cluster_range = (1, max_cluster_number)
        self.n_cluster = self._set_cluster_number()
        self.cluster = [Cluster() for i in range(self.n_cluster)]
        self.df = self._get_df()
        self.X = self.df[["x", "y"]].copy().to_numpy()
    
    def _get_df(self):
        df_list = []
        for i, cluster in enumerate(self.cluster):
            df = cluster.coordinates.copy()
            df["cluster"] = i
            df_list.append(df)
        return pd.concat(df_list)

    def _set_cluster_number(self):
        possible_cluster_numbers = [i for i in range(self.n_cluster_range[0], self.n_cluster_range[1]+1)]
        return random.choice(possible_cluster_numbers)
