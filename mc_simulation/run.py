import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed

from data_simulation.simulate_data import Experiment
from cluster_initialization.init_class import Cluster_initialization
from em_algorithm.em_class import EM

def run_MC_simulation(
    N_experiments = 10, 
    parallel_params={}, 
    init_routines=["random_inside"], 
    experiment_params={}, 
    em_params = {}, 
    cluster_init_params={}
    ):
        
    simulated_experiments = [Experiment(**experiment_params) for i in range(N_experiments)]
    results = {}
    for init_routine in init_routines:
        parallel = Parallel(**parallel_params) 
        parallel_results = parallel(delayed(cluster_experiment)(init_routine, cluster_init_params, em_params, df_experiment=experiment.df) for experiment in simulated_experiments)
        results[init_routine] = merge_parallel_results(parallel_results)
        print("Init_routine run has successfully finished")

    merged_results = merge_init_routine_results(results)
    df_results, model_data = create_df_from_results(merged_results)
    model_data["datasets"] = [experiment.df for experiment in simulated_experiments]
    model_data["distributions"] = [experiment.get_distr_params_df() for experiment in simulated_experiments]
    df_results = include_true_clusternumber(df_results, model_data)
    return df_results, model_data

def include_true_clusternumber(df_results, model_data):
    for dataset in df_results.dataset.unique():
        idx = df_results.dataset == dataset
        df_results.loc[idx, "True_N_Cluster"] = len(model_data["datasets"][dataset].cluster.unique())
    df_results["True_N_Cluster"] = df_results["True_N_Cluster"].astype(int)
    return df_results


def cluster_experiment(init_routine, cluster_init_params, em_params, experiment_params=None, df_experiment=None):
    if experiment_params and df_experiment is None:                         # simulates experimental data
        df_experiment = Experiment(**experiment_params).df                        
    elif type(df_experiment)==pd.DataFrame and experiment_params is None:   # passing experimental data as pd.dataframe[["x", "y"]]
        pass
    else: 
        raise ValueError("You have to pass either experiment_params or experiment.")

    cluster_init = Cluster_initialization(df_experiment, routine=init_routine)
    cluster_init.sample(**cluster_init_params)
    em = EM(**em_params)
    em.run(df_experiment, cluster_init=cluster_init)
    return em.results.get_dict()


def merge_parallel_results(parallel_results):
    merged_results = {key: [] for key in parallel_results[0].keys()}
    merged_results["dataset"] = []
    for dataset_idx, result in enumerate(parallel_results):
        merged_results["dataset"] = merged_results["dataset"] + [dataset_idx for _i in range(len(result["iterations"]))]
        for feature_name, list_ in result.items():
            merged_results[feature_name] = merged_results[feature_name] + list_
    return merged_results

def merge_init_routine_results(results):
    merged_results = {key: [] for key in results[list(results.keys())[0]].keys()}
    merged_results["init_routine"] = []
    for init_routine, result in results.items():
        merged_results["init_routine"] = merged_results["init_routine"] + [init_routine for _i in range(len(results[init_routine]["iterations"]))]
        for key, val in result.items():
            merged_results[key] = merged_results[key] + val
    return merged_results


def create_df_from_results(results, non_df_cols = ["inferred_mixtures", "starting_values"]):
    df_result = pd.DataFrame.from_dict({key: val for key, val in results.items() if key not in non_df_cols})
    df_result = df_result.reset_index().rename(columns={"index": "model_idx"})
    return df_result, {key: val for key, val in results.items() if key in non_df_cols}