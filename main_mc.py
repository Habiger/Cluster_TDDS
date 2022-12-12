import datetime
import os

from clustering.input_data.simulated_experiment_class import SimulatedExperiment, SimulatedExperimentParameter
from clustering.clustering_class import EMClustering

def create_result_dir():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mc_run_results_path = os.path.join("3_mc_results", f"MC_run_{now}")
    if not os.path.exists(mc_run_results_path):
        os.makedirs(mc_run_results_path)
    return mc_run_results_path

def main():
    mc_run_results_path = create_result_dir()
    n_experiments = 100
    exp_params = {
        "store_data": True,
        "path_to_store_data": os.path.join(mc_run_results_path, "datasets"),
        "min_datapoints": 5,
        "cluster_number_range": (1, 10),
        "cluster": {
            "cluster_size_range": (1, 100),
            "exp_scale_range": (0.001, 1),
            "norm_loc_range": (0.5, 7),
            "norm_scale_range": (0.1, 1),
        },
    }

    experiment_data = [SimulatedExperiment(str(i), **exp_params) for i in range(n_experiments)]

    for em_tol in [1e-02, 1e-3, 1e-4, 1e-5, 1e-6]:
        mc_params = {
            "save_data": True,
            "path_to_save": mc_run_results_path,
            "parallel": {"n_jobs": 12, "verbose": 11},
            "cluster_init": {
                "init_routine": "random_inside",
                "routine": {
                    "N_cluster_min": 1,
                    "N_cluster_max": 13,
                    "N_runs_per_clusternumber": 20,
                    "N_max": 10000,
                    "y_scale_range": (0.1, 2),
                },
            },
            "em": {
                "max_reiterations": 500,
                "em_algorithm": {
                    "em_tol": em_tol,
                    "max_iter": 10000,
                    "min_mix_coef": 0.00,
                    "abs_tol_params": 1e-08,
                    "minimizer_options": {"maxiter": 100},
                },
            },
        }

    mc = EMClustering(**mc_params)
    mc.load_experiments(experiment_data)
    mc.save_results()