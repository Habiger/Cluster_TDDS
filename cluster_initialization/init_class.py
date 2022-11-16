from copy import deepcopy
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict
import matplotlib.pyplot as plt

from cluster_initialization.parameter_dataclass import Params

from cluster_initialization.algorithms.rnd import RANDOM_routine
from cluster_initialization.algorithms.rnd_inside import RANDOM_inside_routine
from cluster_initialization.algorithms.optics import OPTICS_routine
from cluster_initialization.algorithms.optics_weighted import OPTICS_weighted_routine




class Routine:
    random = "random"
    random_inside = "random_inside"
    optics = "OPTICS"
    optics_weighted = "OPTICS_weighted"
    all = [random, random_inside, optics, optics_weighted]


class Cluster_initialization:
    routines = {                                         #TODO make an algotihm class where all class specific funcitons, attributes are stored
        Routine.optics: OPTICS_routine,
        Routine.optics_weighted: OPTICS_weighted_routine,
        Routine.random: RANDOM_routine,
        Routine.random_inside: RANDOM_inside_routine
        }

    def __init__(self, df: pd.DataFrame, routine=Routine.optics):
        self.df = df.copy()
        self.variant = routine
        self.Routine= self.routines[routine]

        # set by self._run_algorithm
        self.init_params = None                 
        self.init_params_array = None
        self._run_algorithm()

        # set by self._sample
        self.sampled_init_params = None         
        self.sampled_init_params_array = None   

    def _run_algorithm(self):
        init_params, labels = self.Routine.algorithm(self.df)
        self.init_params = init_params
        self.init_params_array = self._create_param_array(init_params)
        if labels is not None:
            self.df[f"init_cluster"] = labels
    
    @staticmethod
    def _create_param_array(init_params: Dict[int, Params]) -> np.ndarray:
        K = len(init_params.keys())
        param_array = np.ndarray((4 * K))
        for i, cl in enumerate(init_params.keys()):
            param_array[4*i] = init_params[cl].mix_coef
            param_array[1 + 4*i] = init_params[cl].x.mu
            param_array[2 + 4*i] = init_params[cl].y.mu
            param_array[3 + 4*i] = init_params[cl].y.std
        return param_array

    def sample(self, N_cluster_max=10, N_runs_per_clusternumber=10):
        """Does sample a certain amount of starting values from the total set of generated values `self.init_params` by the specific algorithm

        Args:
            * N_cluster_max (int, optional): up to which number of clusters to sample. Defaults to 10.
            * N_runs_per_clusternumber (int, optional): how many sets per clusternumber. Defaults to 10.
        """
        sampled_init_params = self.Routine.get_sampled_init_params(self.df, self.init_params, N_cluster_max, N_runs_per_clusternumber)
        sampled_init_params = self._correct_mix_coef(sampled_init_params)
        self.sampled_init_params = sampled_init_params
        self.sampled_init_params_array = [self._create_param_array(params) for params in sampled_init_params]

    def _correct_mix_coef(self, selected_init_params: Dict[int, Params]) -> Dict[int, Params]:
        selected_init_params_new = []
        for initialization in selected_init_params:
            denominator = sum([pars.mix_coef for cl, pars in initialization.items()])
            corrected_param_dict = {}
            for cl, pars in initialization.items():
                corrected_param_dict[cl] = deepcopy(pars)
                corrected_param_dict[cl].mix_coef = pars.mix_coef/denominator
            selected_init_params_new.append(corrected_param_dict)
        return selected_init_params_new

    def replace_init_param(self, init_param, idx):
        """should be used to replace the atttributes for consistency reasons
        """
        pass
        
    def sample_new_starting_values(self, idx):
        misbehaving_starting_values = self.sampled_init_params_array[idx]
        K = len(misbehaving_starting_values)//4
        init_param_new = self.Routine.get_single_init_param_sample(self.init_params, K)
        init_param_new = self._correct_mix_coef([init_param_new])
        init_param_array_new = self._create_param_array(init_param_new[0])
        self.sampled_init_params_array[idx] = init_param_array_new[:]
        return init_param_array_new

    def plot_initial_values(self, idx):    
        plt.scatter(self.df.x, self.df.y, alpha=0.3, label="Experiment Data")
        fig = plt.gcf()
        fig.set_size_inches(20,10)
        plt.scatter(self.sampled_init_params_array[idx][1::4], self.sampled_init_params_array[idx][2::4], marker="x", s=200, label="Starting values")
        plt.title(f"Starting values derived from initialization Routine  (Index = {idx})", color="w", size=30)
        plt.legend(prop={'size': 20})
        plt.close()
        return fig

