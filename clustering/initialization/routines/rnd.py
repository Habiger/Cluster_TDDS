import pandas as pd
import numpy as np
from typing import Dict
from random import choices

from clustering.initialization.routines.distribution_dataclass import MixtureComponentDistribution




class RandomRoutine:
    """currently not working
    """
    assigns_labels = False
    N_max = 1000

    @classmethod
    def algorithm(cls, df: pd.DataFrame) -> Dict[int, MixtureComponentDistribution]:
        
        logx_min, logx_max = np.log(np.min(df.x)), np.log(np.max(df.x))
        y_min, y_max = np.min(df.y), np.max(df.y)

        params_dict = {}
        for cl in range(cls.N_max):
            params = MixtureComponentDistribution(cl)
            params.x.mu = np.exp(np.random.uniform(logx_min, logx_max))
            params.y.mu = np.random.uniform(y_min, y_max)
            params.y.std =  np.random.uniform(0.1, 1)
            params.mix_coef = 1
            params_dict[cl] = params
        return params_dict, None   # None indicates that the algorithm returns no labels for the original datapoints

    @classmethod
    def get_sampled_init_params(cls, _df, init_params, K_max, N_runs_per_clusternumber):
        cluster_keys = list(init_params.keys())
        sampled_params = []
        for K in range(1, K_max+1):
            for _n in range(0, N_runs_per_clusternumber):
                idx = np.random.choice(cluster_keys, size=K, replace=False)
                sampled_params.append({key: val for key, val in init_params.items() if key in idx})
        return sampled_params