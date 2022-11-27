import numpy as np
import pandas as pd
from dataclasses import dataclass

from miscellaneous.parameter import Parameter, nested_dataclass
from em_algorithm.em_funcs import run_EM, compute_loglikelihood
from cluster_initialization.init_class import Cluster_initialization

@dataclass
class Minimizer_Parameter(Parameter):
    maxiter: int = 100
@nested_dataclass
class EM_Algorithm_Parameter(Parameter):
    em_tol: float = 1e-5    
    max_iter: int = 1000
    min_mix_coef: float = 0.02
    abs_tol_params: float = 1e-8
    minimizer_options: Minimizer_Parameter = Minimizer_Parameter()
    
@nested_dataclass
class EM_Parameter(Parameter):
    """default parameters for em-algorithm
    """
    max_reiterations: int = 50  # number of times em algorithm should be rerun with different values when divergence occurs
    em_algorithm: EM_Algorithm_Parameter = EM_Algorithm_Parameter()


class Results:
    def __init__(self) -> None:
        self.inferred_mixtures = []
        self.iterations = []
        self.execution_time = []
        self.starting_values = []
        self.total_execution_time = []
        self.total_iterations = []
        self.em_model_idx = []
        self.reiterations = []
    
    def update(self, mixtures, iterations, execution_time, total_iterations, total_execution_time, starting_values, model_idx, reiterations):
        self.iterations.append(iterations)
        self.inferred_mixtures.append(mixtures)
        self.execution_time.append(execution_time)
        self.total_iterations.append(total_iterations)
        self.total_execution_time.append(total_execution_time)
        self.starting_values.append(starting_values)
        self.em_model_idx.append(model_idx)
        self.reiterations.append(reiterations)

    def get_dict(self) -> dict:
        return self.__dict__


class EM():
    def __init__(self, **kwargs) -> None:
        self.params = EM_Parameter(**kwargs)

        # will be set by run method
        self.results = Results()

    def run(self, df_experiment:pd.DataFrame, init_param_array: np.ndarray = None, cluster_init: Cluster_initialization = None):
        """Performs EM-Algorithm for either a single model (init_param_array) or a whole model selection (cluster_init). 

        Args:
            df_experiment (pd.DataFrame): dataframe with columns "x" and "y" as the coordinates of datapoints
            init_param_array (np.ndarray, optional): mixture starting values. Defaults to None.
            cluster_init (Cluster_initialization, optional): Cluster Initialization object for model selection run. Defaults to None.

        Raises:
            ValueError: You have to pass one and only one of the two optional parameters. 
        """
        X = df_experiment[["x", "y"]].to_numpy(copy=True)
        if init_param_array and cluster_init is None:
            self._run_from_init_param_array(X, init_param_array)
        elif cluster_init and init_param_array is None:
            self._run_from_Cluster_Init(X, cluster_init)
        else:
            raise ValueError("You have to pass either an init_param_array (single run) or a Cluster_Initialization object (model selection run).")


    def _run_from_init_param_array(self, X: np.ndarray, init_param_array: np.ndarray):
        (mixtures, iterations), execution_time = run_EM(X, init_param_array, **self.params.em_algorithm.get_dict())
        return (mixtures, iterations), execution_time


    def _run_from_Cluster_Init(self, X: np.ndarray, cluster_init: Cluster_initialization):
        for em_model_idx, initial_values_array in enumerate(cluster_init.sampled_starting_values_arrays):
            (mixtures, iterations), execution_time = run_EM(X, initial_values_array, **self.params.em_algorithm.get_dict())
            total_iterations, total_execution_time = iterations, execution_time
            j, ll = 0, compute_loglikelihood(mixtures, X)
            while j < self.params.max_reiterations and not np.isfinite(ll):
                # repeat em with new starting values
                initial_values_array_new_sampled = cluster_init.sample_new_starting_values(em_model_idx)
                if initial_values_array_new_sampled is None:
                    break # if no new set of starting values can be found the reiteration cycle breaks
                (mixtures, iterations), execution_time = run_EM(X, initial_values_array_new_sampled, **self.params.em_algorithm.get_dict())
                total_iterations += iterations
                total_execution_time += execution_time
                j += 1
                ll = compute_loglikelihood(mixtures, X)
            self.results.update(mixtures, iterations, execution_time, total_iterations, total_execution_time, initial_values_array, em_model_idx, j)


            