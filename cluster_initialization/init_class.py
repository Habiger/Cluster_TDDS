from copy import deepcopy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
import matplotlib.pyplot as plt

from miscellaneous.parameter import Parameter
from cluster_initialization.parameter_dataclass import Params

from cluster_initialization.algorithms.rnd import RANDOM_routine
from cluster_initialization.algorithms.rnd_inside import RANDOM_inside_routine
from cluster_initialization.algorithms.optics import OPTICS_routine
from cluster_initialization.algorithms.optics_weighted import OPTICS_weighted_routine

@dataclass
class Cluster_Initialization_Parameter(Parameter):
    N_cluster_max: int = 10
    N_runs_per_clusternumber: int = 10
    init_routine: str = "random_inside"




class Cluster_initialization:
    routine_algorithms = {                                         #TODO make an algotihm class where all class specific funcitons, attributes are stored
        "OPTICS": OPTICS_routine,
        "OPTICS_weighted": OPTICS_weighted_routine,
        "random": RANDOM_routine,
        "random_inside": RANDOM_inside_routine
        }

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.params = Cluster_Initialization_Parameter(**kwargs)

        self.df = df.copy()
        self.Routine= self.routine_algorithms[self.params.init_routine]

        # set by self._run_algorithm
        self.possible_starting_values = None                 
        self._run_algorithm()

        # set by self._sample
        self.sampled_starting_values_dict = None         
        self.sampled_starting_values_arrays = None   

    def _run_algorithm(self):
        possible_starting_values, labels = self.Routine.algorithm(self.df)
        self.possible_starting_values = possible_starting_values
        if labels is not None:
            self.df[f"init_cluster"] = labels
    
    @staticmethod
    def _create_param_array(starting_value_dict: Dict[int, Params]) -> np.ndarray:
        K = len(starting_value_dict.keys())
        param_array = np.ndarray((4 * K))
        for i, cl in enumerate(starting_value_dict.keys()):
            param_array[4*i] = starting_value_dict[cl].mix_coef
            param_array[1 + 4*i] = starting_value_dict[cl].x.mu
            param_array[2 + 4*i] = starting_value_dict[cl].y.mu
            param_array[3 + 4*i] = starting_value_dict[cl].y.std
        return param_array

    def sample(self):
        """Does sample a certain amount of starting values from the total set of generated values `self.init_params` by the specific algorithm
        """
        sampled_init_params = self.Routine.get_sampled_init_params(self.possible_starting_values, self.params.N_cluster_max, self.params.N_runs_per_clusternumber)
        sampled_init_params = self._correct_mix_coef(sampled_init_params)
        self.sampled_starting_values_dict = sampled_init_params
        self.sampled_starting_values_arrays = [self._create_param_array(params) for params in sampled_init_params]

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

    def sample_new_starting_values(self, idx):
        misbehaving_starting_values = self.sampled_starting_values_arrays[idx]
        K = len(misbehaving_starting_values)//4
        init_param_new = self.Routine.get_single_init_param_sample(self.possible_starting_values, K, self.sampled_starting_values_dict)
        if init_param_new is None:
            return None
        else:
            init_param_new = self._correct_mix_coef([init_param_new])
            init_param_array_new = self._create_param_array(init_param_new[0])
            self.sampled_starting_values_arrays[idx] = init_param_array_new[:] # is it necessary to include newly sampled starting values ?
            return init_param_array_new



    def plot_initial_values(self, idx):    
        plt.scatter(self.df.x, self.df.y, alpha=0.3, label="Experiment Data")
        fig = plt.gcf()
        fig.set_size_inches(20,10)
        plt.scatter(self.sampled_starting_values_arrays[idx][1::4], self.sampled_starting_values_arrays[idx][2::4], marker="x", s=200, label="Starting values")
        plt.title(f"Starting values derived from initialization Routine  (Index = {idx})", color="w", size=30)
        plt.legend(prop={'size': 20})
        plt.close()
        return fig

