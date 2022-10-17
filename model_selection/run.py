import numpy as np

# Parallelization
from joblib import Parallel, delayed
from tqdm import tqdm

from em_algorithm.processing_results import process_parallel_results
from em_algorithm.em_funcs import run_EM
from model_selection.scoreboard import create_scoreboard


def model_selection_run(X, cluster_init, em_params, init_params, parallel_params):
    #TODO - refactor: isolate check fo sigularity ("ll")
    #TODO - check wether other init routines have singularity issues too
    #TODO - organize results better
    # TODO - think about how/wether save results from singular runs (e.g. time)
    # -> maybe store time and old results in cluster_init
    cluster_init.sample(**init_params)
    init_params_arrays = cluster_init.sampled_init_params_array
    parallel_results = Parallel(**parallel_params)(delayed(run_EM)(X, init_param_array, **em_params) for init_param_array in tqdm(init_params_arrays))
    results = process_parallel_results(parallel_results)
    df_scores, df_scores_na = create_scoreboard(results, X) # need to be run because "ll" will calculated thourgh scorerboard
    # rerun em algorithm for singular results
    singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
    while len(singular_result_idxs) > 0:
        print(singular_result_idxs)
        print(np.array(results["iter_steps"])[singular_result_idxs])
        singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
        parallel_rerun_results = Parallel(n_jobs=10)(delayed(run_EM)(X, cluster_init.sample_new_starting_values(idx), **em_params) for idx in tqdm(singular_result_idxs))
        rerun_results = process_parallel_results(parallel_rerun_results)
        for key in rerun_results.keys():
            for i, idx in enumerate(singular_result_idxs):
                results[key][idx] = rerun_results[key][i]
        df_scores, df_scores_na = create_scoreboard(results, X)
        singular_result_idxs = np.where(np.isnan(results["ll"]))[0]
    return df_scores, results