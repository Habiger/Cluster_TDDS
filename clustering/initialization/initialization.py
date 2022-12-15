import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from dataclasses import field

from clustering.miscellaneous.parameter_dataclass import Parameter, nested_dataclass

from clustering.initialization.routines.distribution_dataclass import MixtureComponentDistribution
from clustering.initialization.routines.base_classes import RoutineParameter, Routine
from clustering.initialization.routines.rnd_inside import RandomInsideRoutine, RandomInsideRoutineParameter
from clustering.initialization.routines.optics import OPTICSRoutine, OPTICSRoutineParameter
from clustering.initialization.routines.rnd import RandomRoutine
from clustering.initialization.routines.optics_weighted import OPTICSWeightedRoutine


@nested_dataclass
class InitializationParameter(Parameter):
    init_routine: str = "random_inside"
    routine: RoutineParameter = field(init=False)   # will be filled according to `init_routine` during the `self.__post_init__()` call

    def __post_init__(self, **kwargs):              # kwargs will be passed by decorator `nested_dataclass`
        if self.init_routine == "random_inside":
            self.routine: RandomInsideRoutineParameter =  RandomInsideRoutineParameter(**kwargs)
        elif self.init_routine == "OPTICS":
            self.routine: OPTICSRoutineParameter =  OPTICSRoutineParameter(**kwargs)


class Initialization:
    routine_algorithms = {                                    
        "OPTICS": OPTICSRoutine,
        "random_inside": RandomInsideRoutine,
        #"OPTICS_weighted": OPTICSWeightedRoutine,  # does currently not work
        #"random": RandomRoutine                    # does currently not work
        }

    def __init__(self, X: np.ndarray, **kwargs):
        self.params = InitializationParameter(**kwargs)

        self.X = X.copy()
        self.df = pd.DataFrame.from_dict({"x": X[:, 0], "y": X[:, 1]})
        self.Routine: Routine = self.routine_algorithms[self.params.init_routine](**self.params.routine.get_dict())

        # set by self._run_algorithm()
        self.possible_starting_values = None                 
        self._run_algorithm()

        # set by self.sample()
        self.sampled_starting_values_dict = None         
        self.sampled_starting_values_arrays = None   

    def _run_algorithm(self):
        possible_starting_values, labels = self.Routine.algorithm(self.X)
        self.possible_starting_values = possible_starting_values
        if labels is not None:
            self.df[f"init_cluster"] = labels

    def sample(self):
        """Does sample a certain amount of starting values from the total set of generated values `self.init_params` by the specific algorithm
        """
        sampled_init_params = self.Routine.get_sampled_init_params(self.possible_starting_values)
        sampled_init_params = self._correct_mix_coef(sampled_init_params)
        self.sampled_starting_values_dict = sampled_init_params
        self.sampled_starting_values_arrays = [self._create_param_array(params) for params in sampled_init_params]

    def _correct_mix_coef(self, selected_starting_values: dict[int, MixtureComponentDistribution]) -> dict[int, MixtureComponentDistribution]:
        selected_init_params_new = []
        for initialization in selected_starting_values:
            denominator = sum([pars.mix_coef for cl, pars in initialization.items()])
            corrected_param_dict = {}
            for cl, pars in initialization.items():
                corrected_param_dict[cl] = deepcopy(pars)
                corrected_param_dict[cl].mix_coef = pars.mix_coef/denominator
            selected_init_params_new.append(corrected_param_dict)
        return selected_init_params_new

    def sample_new_starting_values(self, idx: int) -> None | np.ndarray:
        misbehaving_starting_values = self.sampled_starting_values_arrays[idx]
        K = len(misbehaving_starting_values)//4
        init_param_new = self.Routine.get_single_init_param_sample(self.possible_starting_values, K, self.sampled_starting_values_dict)
        if init_param_new is None:
            return None
        else:
            init_param_new = self._correct_mix_coef([init_param_new])
            init_param_array_new = self._create_param_array(init_param_new[0])
            self.sampled_starting_values_arrays[idx] = init_param_array_new[:] #TODO is it necessary to include newly sampled starting values ?
            return init_param_array_new

    @staticmethod
    def _create_param_array(starting_value_dict: dict[int, MixtureComponentDistribution]) -> np.ndarray:
        K = len(starting_value_dict.keys())
        param_array = np.ndarray((4 * K))
        for i, cl in enumerate(starting_value_dict.keys()):
            param_array[4*i] = starting_value_dict[cl].mix_coef
            param_array[1 + 4*i] = starting_value_dict[cl].x.mu
            param_array[2 + 4*i] = starting_value_dict[cl].y.mu
            param_array[3 + 4*i] = starting_value_dict[cl].y.std
        return param_array


    def plot_initial_values(self, idx: int):    
        plt.scatter(self.X.x, self.X.y, alpha=0.3, label="Experiment Data")
        fig = plt.gcf()
        fig.set_size_inches(20,10)
        plt.scatter(self.sampled_starting_values_arrays[idx][1::4], self.sampled_starting_values_arrays[idx][2::4], marker="x", s=200, label="Starting values")
        plt.title(f"Starting values derived from initialization Routine  (Index = {idx})", color="w", size=30)
        plt.legend(prop={'size': 20})
        plt.close()
        return fig

