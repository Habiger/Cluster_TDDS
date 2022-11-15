import numpy as np
from copy import deepcopy
# Parallelization
from joblib import Parallel, delayed
from tqdm import tqdm

from em_algorithm.processing_results import process_parallel_results
from em_algorithm.em_funcs import run_EM
from model_selection.scoreboard import create_scoreboard, update_results_ll


def model_selection_run_old(X, cluster_init, em_params, init_params, parallel_params):
    #TODO - refactor: isolate check fo sigularity ("ll")
    #TODO - check wether other init routines have singularity issues too
    #TODO - organize results better
    # TODO - think about how/wether save results from singular runs (e.g. time)
    # -> maybe store time and old results in cluster_init
    cluster_init.sample(**init_params)
    init_params_arrays = cluster_init.sampled_init_params_array
    parallel_results = Parallel(**parallel_params)(delayed(run_EM)(X, init_param_array, **em_params) for init_param_array in init_params_arrays)
    results = process_parallel_results(parallel_results)
    df_scores, df_scores_na = create_scoreboard(results, X) # need to be run because "ll" will calculated thourgh scorerboard
    # rerun em algorithm for singular results
    singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
    i = 0
    while (len(singular_result_idxs) > 0) and (i < 50):
        singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
        parallel_rerun_results = Parallel(n_jobs=10)(delayed(run_EM)(X, cluster_init.sample_new_starting_values(idx), **em_params) for idx in singular_result_idxs)
        rerun_results = process_parallel_results(parallel_rerun_results)
        for key in rerun_results.keys():
            for i, idx in enumerate(singular_result_idxs):
                results[key][idx] = rerun_results[key][i]
        df_scores, df_scores_na = create_scoreboard(results, X)
        singular_result_idxs, i = np.where(np.isnan(results["ll"]))[0], i+1
    return df_scores, results

def model_selection_run(X, cluster_init, em_params, init_params, parallel_params):
    cluster_init.sample(**init_params)
    init_params_arrays = cluster_init.sampled_init_params_array
    parallel_results = Parallel(**parallel_params)(delayed(run_EM)(X, init_param_array, **em_params) for init_param_array in init_params_arrays)
    results = process_parallel_results(parallel_results)
    results = update_results_ll(results, X)   #df_scores, df_scores_na = create_scoreboard(results, X) # need to be run because "ll" will calculated thourgh scorerboard
    results["total_execution_time"] = deepcopy(results["execution_time"])
    results["count_repeats"] = [0 for i in range(len(results["execution_time"]))]
    results["total_iter_steps"] = deepcopy(results["iter_steps"])
    # rerun em algorithm for singular results
    singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
    j = 0
    while (len(singular_result_idxs) > 0) and (j < 50):
        singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
        parallel_rerun_results = Parallel(n_jobs=10)(delayed(run_EM)(X, cluster_init.sample_new_starting_values(idx), **em_params) for idx in singular_result_idxs)
        rerun_results = process_parallel_results(parallel_rerun_results)
        for i, idx in enumerate(singular_result_idxs):
            for key in rerun_results.keys():
                results[key][idx] = rerun_results[key][i]
            results["total_execution_time"][idx] += rerun_results["execution_time"][i]
            results["count_repeats"][idx] += 1
            results["total_iter_steps"][idx] += rerun_results["iter_steps"][i]
        results = update_results_ll(results, X)
        singular_result_idxs, j = np.where(np.isnan(results["ll"]))[0], j+1
    df_scores, df_scores_na = create_scoreboard(results, X)
    return df_scores, results
