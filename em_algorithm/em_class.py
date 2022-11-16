from em_algorithm.em_funcs import run_EM, compute_loglikelihood
from cluster_initialization.init_class import Cluster_initialization

class Parameter:
    """default parameters for em-algorithm
    """
    em_tol = 1e-5
    max_iter = 1000
    min_mix_coef = 0.02
    abs_tol_params = 1e-8
    minimizer_options = {
        "maxiter": 100
    } 
    max_reiterations = 50  # number of times em algorithm should be rerun with different values when divergence occurs

import numpy as np

class Results:
    def __init__(self) -> None:
        self.inferred_mixtures = []
        self.iterations = []
        self.execution_time = []
        self.starting_values = []
        self.total_execution_time = []
        self.total_iterations = []
        self.model_idxs = []
        self.reiterations = []
    
    def update_results(self, mixtures, iterations, execution_time, total_iterations, total_execution_time, starting_values, model_idx, reiterations):
        self.iterations.append(iterations)
        self.inferred_mixtures.append(mixtures)
        self.execution_time.append(execution_time)
        self.total_iterations.append(total_iterations)
        self.total_execution_time.append(total_execution_time)
        self.starting_values.append(starting_values)
        self.model_idxs.append(model_idx)
        self.reiterations.append(reiterations)

    def get_dict(self) -> dict:
        return self.__dict__


class EM(Parameter):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)        # sets manually specified Parameters
        self.em_params = self.get_em_params()

        # will be set by run method
        self.results = Results()

    def get_em_params(self):
        em_params = {
            "em_tol": self.em_tol,
            "max_iter": self.max_iter,
            "min_mix_coef": self.min_mix_coef,
            "abs_tol_params": self.abs_tol_params,
            "minimizer_options": self.minimizer_options
        }
        return em_params


    def run(self, X, init_param_array: np.ndarray = None, cluster_init: Cluster_initialization = None):
        """Performs EM-Algorithm for either a single model (init_param_array) or a whole model selection (cluster_init). 

        Args:
            X (np.ndarray): x and y coordinates of datapoints
            init_param_array (np.ndarray, optional): mixture starting values. Defaults to None.
            cluster_init (Cluster_initialization, optional): Cluster Initialization object for model selection run. Defaults to None.

        Raises:
            ValueError: You have to pass one and only one of the two optional parameters. 
        """
        if init_param_array and cluster_init is None:
            self._run_from_init_param_array(X, init_param_array)
        elif cluster_init and init_param_array is None:
            self._run_from_Cluster_Init(X, cluster_init)
        else:
            raise ValueError("You have to pass either an init_param_array (single run) or a Cluster_Initialization object (model selection run).")


    def _run_from_init_param_array(self, X: np.ndarray, init_param_array: np.ndarray):
        (mixtures, iterations), execution_time = run_EM(X, init_param_array, **self.em_params)
        return (mixtures, iterations), execution_time


    def _run_from_Cluster_Init(self, X: np.ndarray, cluster_init: Cluster_initialization):
        for model_idx, init_param_array in enumerate(cluster_init.sampled_init_params_array):
            (mixtures, iterations), execution_time = run_EM(X, init_param_array, **self.em_params)
            total_iterations, total_execution_time = iterations, execution_time
            j, ll = 0, compute_loglikelihood(mixtures, X)
            while j < self.max_reiterations and not np.isfinite(ll):
                if j != 0:
                    init_param_array = cluster_init.sample_new_starting_values(model_idx)
                (mixtures, iterations), execution_time = run_EM(X, init_param_array, **self.em_params)
                total_iterations += iterations
                total_execution_time += execution_time
                j += 1
                ll = compute_loglikelihood(mixtures, X)
            self.results.update_results(mixtures, iterations, execution_time, total_iterations, total_execution_time, init_param_array, model_idx, j)


            