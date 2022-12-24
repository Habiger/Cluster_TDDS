import os
import pandas as pd
import datetime

from dataclasses import dataclass
from joblib import Parallel, delayed

from clustering.miscellaneous.parameter_dataclass import Parameter, nested_dataclass
from clustering.miscellaneous.exception_handler_decorator import catch_exceptions
from clustering.miscellaneous.logger import Logger
from clustering.miscellaneous.save_json import save_dict_as_json

from clustering.input_data.experiment_baseclass import Experiment
from clustering.initialization.initialization import Initialization, InitializationParameter
from clustering.em.em_class import EM, EMParameter
from clustering.model_selection.criteria.scoreboard import create_scoreboard

@dataclass
class ParallelParameter(Parameter):
    n_jobs: int = 10                                                    # number of parallel processes
    verbose: int = 11                                                   # how much progress information is shown during execution, see also joblib parallel


@nested_dataclass
class EMClusteringParameter(Parameter):
    path_to_store_data: str = None                                            # (see above)
    parallel: ParallelParameter = ParallelParameter()                   # parameters passed to joblib´s `Parallel`
    cluster_init: InitializationParameter = InitializationParameter()   # parameters passed to Initialization class
    em: EMParameter = EMParameter()                                     # parameters passed to class `EM`
    

class EMClustering:
    """Runs the clustering for a list of experimental data in parallel. \\
        * Change default parameters by passing appropriate `kwargs` to `__init__`. 
        * You have to pass a list of `Experiments` via `self.load_experiment()`. 
        * Then you can run the `self.run()` method to obtain the results which will be 
        stored in `self.model_data` and `self.df_results`.
    """ 
    logger = Logger.start_logger_if_necessary()

    def __init__(self, **kwargs) -> None:
        self.params = EMClusteringParameter(**kwargs) 
        self.logger.info(
            f"Initialization of a MC object with the following parameters has been carried out.\n{self.params}"
            )
        # processed results of the clustering run:
        self.model_data = None
        self.df_scores = None
        self.df_scores_na = None

    def load_experiments(self, experiments: list[Experiment]):
        """Load experiments; look at `Experiment` class docstring for further information.

        Args:
            experiments (list[Experiment]): list of `Experiment` instances
        """
        self.experiments = experiments
        self.N_experiments = len(experiments)
    

    @catch_exceptions(logger)
    def run(self):
        """Calling this method will start the parallelized clustering.\\
            Make sure you have provided the experimental data through `self.load_experiments()`.\\
            The outcome will be stored in `self.model_data` and `self.df_results`
        """
        self.logger.info(
            "Starting the clustering of experimental data batch" +
            f"with init_routine {self.params.cluster_init.init_routine}.\n"
            )
        parallel_results = self._execute_parallel_clustering()

        self.logger.info("\nClustering successfull; starting processing of results")
        df_results, self.model_data = self._process_parallel_results(parallel_results)
        self.df_scores, self.df_scores_na  = create_scoreboard(df_results, self.model_data)
        self.logger.info(
            "Processing successfull - Run has been finished - logger will be closed - END\n\n"
            )
        Logger.close_logger()


    @catch_exceptions(logger)
    def _execute_parallel_clustering(self) -> list:
        """Uses `joblib` library to perform the clustering of each experiment in parallel.\\
            Each process will call `self._cluster_experiment()`.
        """
        parallel = Parallel(**self.params.parallel.get_dict()) 
        parallel_results = parallel(delayed(self._cluster_experiment)(
            experiment,
            self.params.cluster_init.get_dict(), 
            self.params.em.get_dict()) for experiment in self.experiments
        )
        self.logger.info(
            f"Clustering with init routine {self.params.cluster_init.init_routine} has finished."
            )
        return parallel_results

    @staticmethod   # joblib can only handle functions/staticmethods
    @catch_exceptions(logger)
    def _cluster_experiment(experiment: Experiment, cluster_init_params: dict, em_params: dict) -> dict:
        """Function that is called by each individual parallel process.

        Args:
            * experiment (Experiment): contains experimental data and its corresponding ID
            * cluster_init_params (dict): parameters passed to `Cluster_Initialization`
            * em_params (dict): parameters passed to `EM`

        Returns:
            (dict): contains stats and results from the executed EM-algorithm
        """
        X = experiment.X.copy()
        logger = Logger.start_logger_if_necessary()  # needed because joblib sometimes has no access to logger
        cluster_init = Initialization(X, **cluster_init_params)
        cluster_init.sample()
        em = EM(**em_params)
        em.run(X, cluster_init=cluster_init)
        logger.info(f"Experiment {experiment.id} has finished.")
        return em.results.get_dict()

    
    @catch_exceptions(logger)
    def _process_parallel_results(self, parallel_results):
        """wrapper for processing results generated by `self._parallel_clustering()`
        """
        merged_results = self._merge_parallel_results(parallel_results)
        df_results, model_data = self._create_df_from_results(merged_results) 
        df_results = self._include_model_clusternumber(df_results, model_data)

        model_data["datasets"] = {experiment.id: experiment.df for experiment in self.experiments}
        model_data["init_routine"] = self.params.cluster_init.init_routine
        return df_results, model_data


    @catch_exceptions(logger)
    def _merge_parallel_results(self, parallel_results):
        merged_result = {key: [] for key in parallel_results[0].keys()}
        merged_result["dataset"] = []
        for experiment, result in zip(self.experiments, parallel_results):
            if result is None:
                self.logger.warning(f"Custering of dataset {experiment.id} has failed.")
                continue
            merged_result["dataset"] = merged_result["dataset"] + [experiment.id for _i in range(len(result["iterations"]))]
            for feature_name, list_ in result.items():
                merged_result[feature_name] = merged_result[feature_name] + list_
        return merged_result
    
    @catch_exceptions(logger)
    def _create_df_from_results(self, merged_results, non_df_cols = ["inferred_mixtures", "starting_values"]):
        df_result = pd.DataFrame.from_dict({key: val for key, val in merged_results.items() if key not in non_df_cols})
        df_result = df_result.reset_index().rename(columns={"index": "model_idx"}) 
        return df_result, {key: val for key, val in merged_results.items() if key in non_df_cols}


    @staticmethod 
    def _include_model_clusternumber(df_results, model_data):
        N_cluster_list = []
        for model in df_results.itertuples():
            N_cluster_list.append(len(model_data["inferred_mixtures"][model.model_idx])//4)
        df_results["N_cluster"] = N_cluster_list
        return df_results

    def select_best_models(self): #TODO
        """determine best model according to model_selection
        """
        pass

    def save_results(self):
        """saves all relevant data if
        * self.params.path_to_store_data has been set to a valid path
        """
        if self.params.path_to_store_data is None:
            raise ValueError("either `store_data` is set to False or no path for saving data has been provided")
        
          # make directory
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clustering_results_path = os.path.join(self.params.path_to_store_data, f"Clustering_run_{now}")
        if not os.path.exists(clustering_results_path):
            os.makedirs(clustering_results_path)

        # save inferred mixtures
        inferred_mixtures = {
            model_idx: list(inf_mixture) for model_idx, inf_mixture in enumerate(self.model_data["inferred_mixtures"])
            }
        save_dict_as_json(clustering_results_path, inferred_mixtures, "inferred_mixtures.json")

        # save starting values
        starting_values = {
            model_idx: list(starting_values) for model_idx, starting_values in enumerate(self.model_data["starting_values"])
            }
        save_dict_as_json(clustering_results_path, starting_values, "starting_values.json")

        # save df_scores
        filepath_df_scores = os.path.join(clustering_results_path, "df_scores.csv")
        self.df_scores.to_csv(filepath_df_scores, index=False)

        # save df_scores_na
        filepath_df_scores_na = os.path.join(clustering_results_path, "df_scores_na.csv")
        self.df_scores_na.to_csv(filepath_df_scores_na, index=False)
        
        # save parameters
        self.params.save(clustering_results_path)
