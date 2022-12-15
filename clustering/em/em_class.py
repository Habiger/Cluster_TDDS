import numpy as np
import pandas as pd
from dataclasses import dataclass

from clustering.miscellaneous.parameter_dataclass import Parameter, nested_dataclass
from clustering.em.em_algorithm import run_EM, compute_loglikelihood
from clustering.initialization.initialization import Initialization

@dataclass
class MinimizerParameter(Parameter):
    """options for the scipy minimizer used in the EM implementation
    """
    maxiter: int = 100                                             # maximum number of iterations in one minimizer call

@nested_dataclass
class EMAlgorithmParameter(Parameter):
    """Parameters for the EM function `run_EM()`
    """                                                            # The algorithm stops when ...
    em_tol: float = 1e-5                                           # ... the difference between two consecutive log likelihoods are smaller than this value or
    max_iter: int = 1000                                           # ... the maximum number of iterations has been reached. 
    min_mix_coef: float = 0.02                                     # Minimizer-Constraint:  minimum value for the mixture component coefficient; helps by reducing divergence
    abs_tol_params: float = 1e-8                                   # Minimizer-Constraint:  Distribution parameters must be bigger than this value.
    minimizer_options: MinimizerParameter = MinimizerParameter()   # Options passed to scipy minimizer
    
@nested_dataclass
class EMParameter(Parameter):
    """Parameters for the `EM`class.
    """
    max_reiterations: int = 50                                     # number of times em algorithm should be rerun with different values when divergence occurs
    em_algorithm: EMAlgorithmParameter = EMAlgorithmParameter()


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
    
    def update(self, mixtures, iterations, execution_time, total_iterations, total_execution_time, starting_values, em_model_idx, reiterations):
        self.iterations.append(iterations)
        self.inferred_mixtures.append(mixtures)
        self.execution_time.append(execution_time)
        self.total_iterations.append(total_iterations)
        self.total_execution_time.append(total_execution_time)
        self.starting_values.append(starting_values)
        self.em_model_idx.append(em_model_idx)
        self.reiterations.append(reiterations)

    def get_dict(self) -> dict:
        return self.__dict__


class EM:
    """Basically a wrapper around the function `run_EM()`.\\
        Its main purpose is to execute reiterations of the EM algorithm if the initially given starting values lead to a divergent loglikelihood.\\
            The new starting values for the reiterations are provided by an instance of `ClusterInitialization`.
    """
    def __init__(self, **kwargs) -> None:
        self.params = EMParameter(**kwargs)
        self.results = Results()  # will be filled by run method


    def run(self, X: np.ndarray, starting_value_array: np.ndarray = None, cluster_init: Initialization = None):
        """Performs EM-Algorithm for either a single model (init_param_array) or a whole model selection run (cluster_init). 

        Args:
            X (np.ndarray): array with columns "x" and "y"
            starting_value_array (np.ndarray, optional): mixture parameter starting values. Defaults to None.
            cluster_init (Cluster_initialization, optional): Cluster Initialization instance used for (re-)sampling starting values. Defaults to None.

        Raises:
            ValueError: You have to pass one and only one of the two optional parameters. 
        """
        if starting_value_array and cluster_init is None:
            self._run_from_init_param_array(X, starting_value_array)
        elif cluster_init and starting_value_array is None:
            self._run_from_Cluster_Init(X, cluster_init)
        else:
            raise ValueError("You have to pass either an init_param_array (single run) or a Cluster_Initialization object (model selection run).")


    def _run_from_init_param_array(self, X: np.ndarray, init_param_array: np.ndarray):
        (mixtures, iterations), execution_time = run_EM(X, init_param_array, **self.params.em_algorithm.get_dict())
        return (mixtures, iterations), execution_time

    def _run_from_Cluster_Init(self, X: np.ndarray, cluster_init: Initialization):   #TODO eliminate dependency on Initialization by explicitely passing needed things
        for em_model_idx, starting_values_array in enumerate(cluster_init.sampled_starting_values_arrays):
            (mixtures, iterations), execution_time = run_EM(X, starting_values_array, **self.params.em_algorithm.get_dict())
            total_iterations, total_execution_time = iterations, execution_time
            j, ll = 0, compute_loglikelihood(X, mixtures)
            while j < self.params.max_reiterations and not np.isfinite(ll):
                # repeat em with new starting values
                starting_values_array_new_sampled = cluster_init.sample_new_starting_values(em_model_idx)
                if starting_values_array_new_sampled is None:
                    break # if no new set of starting values can be found the reiteration cycle breaks
                (mixtures, iterations), execution_time = run_EM(X, starting_values_array_new_sampled, **self.params.em_algorithm.get_dict())
                total_iterations += iterations
                total_execution_time += execution_time
                j += 1
                ll = compute_loglikelihood(X, mixtures)
            self.results.update(mixtures, iterations, execution_time, total_iterations, total_execution_time, starting_values_array, em_model_idx, j)


            