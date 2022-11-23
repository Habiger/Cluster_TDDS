import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
import logging

from data_simulation.simulate_data import Experiment
from cluster_initialization.init_class import Cluster_initialization
from em_algorithm.em_class import EM
from miscellaneous.exception_handler_decorator import catch_exceptions
from miscellaneous.logger import create_logger, start_logger_if_necessary

class MCDefaultParameter:
    N_experiments = 5

    parallel_params = {
        "n_jobs": 10,
        "verbose": 11,
    }

    init_routines = ["random_inside"]

    experiment_params = {
        "cluster_number_range": (1,5),
        "min_datapoints": 5
    }

    em_params = {
        "max_iter": 500,
        "em_tol": 1e-5,
        "min_mix_coef": 0.02
    }

    cluster_init_params = {
        "N_cluster_max": 7,
        "N_runs_per_clusternumber": 10
    }


class MC(MCDefaultParameter):
    logger = start_logger_if_necessary()

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.parallel_results = None
        self.results = None
        self.merged_parallel_results = None
        self.merged_results = None


    @catch_exceptions(logger)
    def run(self):
        self._simulate_experiments()
        parallel_results = self._cluster_experiments_parallel()
        
        # processing of results
        merged_parallel_results = self._merge_parallel_results(parallel_results)
        merged_results = self._merge_init_routine_results(merged_parallel_results)
        self.df_results, self.model_data = self._create_df_from_results(merged_results) 
        self.include_data_from_true_models(self.df_results, self.model_data)
        # close filehandler of logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def include_data_from_true_models(self):  #TODO refactor
        self.model_data["datasets"] = [experiment.df for experiment in self.simulated_experiments]
        self.model_data["true_models"] = [experiment.get_distr_params_df() for experiment in self.simulated_experiments]
        self.df_results = self.include_true_clusternumber(self.df_results, self.model_data)


    @staticmethod   
    def include_true_clusternumber(df_results, model_data):
        for dataset in df_results.dataset.unique():
            idx = df_results.dataset == dataset
            df_results.loc[idx, "True_N_Cluster"] = len(model_data["datasets"][dataset].cluster.unique())
        df_results["True_N_Cluster"] = df_results["True_N_Cluster"].astype(int)
        return df_results


    
    def _simulate_experiments(self):
        self.simulated_experiments = [Experiment(**self.experiment_params) for i in range(self.N_experiments)]

    @catch_exceptions(logger)
    def _cluster_experiments_parallel(self):
        start_logger_if_necessary()
        parallel_results = {}
        for init_routine in self.init_routines:
            parallel = Parallel(**self.parallel_params) 
            parallel_results[init_routine] = parallel(delayed(self._cluster_experiment)(
                init_routine, 
                self.cluster_init_params, 
                self.em_params, 
                i,
                df_experiment=experiment.df
                ) for i, experiment in enumerate(self.simulated_experiments)
            )
            self.logger.info("Init routine %s has finished.", init_routine, exc_info=1)
        self.parallel_results = parallel_results
        return parallel_results
    
    @staticmethod
    @catch_exceptions(logger)
    def _cluster_experiment(init_routine, cluster_init_params, em_params, i, experiment_params=None, df_experiment=None):
        logger = start_logger_if_necessary()
        if experiment_params and df_experiment is None:                         # simulates experimental data
            df_experiment = Experiment(**experiment_params).df                        
        elif type(df_experiment)==pd.DataFrame and experiment_params is None:   # passing experimental data as pd.dataframe[["x", "y"]]
            pass
        else: 
            raise ValueError("You have to pass either experiment_params or df_experiment.")
        cluster_init = Cluster_initialization(df_experiment, routine=init_routine)
        cluster_init.sample(**cluster_init_params)
        em = EM(**em_params)
        em.run(df_experiment, cluster_init=cluster_init)
        logger.info(f"Experiment {i} has finished.", exc_info=1)
        return em.results.get_dict()

    @catch_exceptions(logger)
    def _merge_parallel_results(self, parallel_results):
        def merge_parallel_result(parallel_result):
            merged_result = {key: [] for key in parallel_result[0].keys()}
            merged_result["dataset"] = []
            for dataset_idx, result in enumerate(parallel_result):
                merged_result["dataset"] = merged_result["dataset"] + [dataset_idx for _i in range(len(result["iterations"]))]
                for feature_name, list_ in result.items():
                    merged_result[feature_name] = merged_result[feature_name] + list_
            return merged_result

        merged_parallel_results = {}
        for init_routine in self.init_routines:
            merged_parallel_results[init_routine] = merge_parallel_result(parallel_results[init_routine])

        self.merged_parallel_results = merged_parallel_results
        return merged_parallel_results

    @catch_exceptions(logger)
    def _merge_init_routine_results(self,merged_parallel_results):
        merged_results = {key: [] for key in merged_parallel_results[list(merged_parallel_results.keys())[0]].keys()}
        merged_results["init_routine"] = []
        for init_routine, result in merged_parallel_results.items():
            merged_results["init_routine"] = merged_results["init_routine"] + [init_routine for _i in range(len(merged_parallel_results[init_routine]["iterations"]))]
            for key, val in result.items():
                merged_results[key] = merged_results[key] + val
        self.merged_results = merged_results
        return merged_results

    @catch_exceptions(logger)
    def _create_df_from_results(self, merged_results, non_df_cols = ["inferred_mixtures", "starting_values"]):
        df_result = pd.DataFrame.from_dict({key: val for key, val in merged_results.items() if key not in non_df_cols})
        df_result = df_result.reset_index().rename(columns={"index": "model_idx"})
        return df_result, {key: val for key, val in merged_results.items() if key in non_df_cols}

