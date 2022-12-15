import datetime
import os

from clustering.input_data.simulated_experiment_class import SimulatedExperiment
from clustering.clustering import EMClustering
def create_result_dir(path_to_store_data = "3_mc_results"):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mc_run_results_path = os.path.join(path_to_store_data, f"MC_run_{now}")
    if not os.path.exists(mc_run_results_path):
        os.makedirs(mc_run_results_path)
    return mc_run_results_path

def simulate_measurement_data(mc_run_results_path, n_experiments):
    exp_params = {
        "store_data": True,
        "path_to_store_data": os.path.join(mc_run_results_path, "datasets"),
        "min_datapoints": 5,
        "cluster_number_range": (1, 5),
        "cluster": {
            "cluster_size_range": (1, 100),
            "exp_scale_range": (0.001, 1),
            "norm_loc_range": (0.5, 7),
            "norm_scale_range": (0.1, 1),
        },
    }
    return [SimulatedExperiment(str(i), **exp_params) for i in range(n_experiments)]


def main():
    mc_run_results_path = create_result_dir()

    n_experiments = 1
    measurement_data = simulate_measurement_data(mc_run_results_path, n_experiments)

    for em_tol in [1e-02]: #[1e-02, 1e-3, 1e-4, 1e-5, 1e-6]:
        mc_params = {
            "path_to_store_data": mc_run_results_path,
            "parallel": {"n_jobs": 12, "verbose": 11},
            "cluster_init": {
                "init_routine": "random_inside",
                "routine": {
                    "N_cluster_min": 1,
                    "N_cluster_max": 5,
                    "N_runs_per_clusternumber": 3,
                    "N_max": 100,
                    "y_scale_range": (0.1, 2),
                },
            },
            "em": {
                "max_reiterations": 500,
                "em_algorithm": {
                    "em_tol": em_tol,
                    "max_iter": 100,
                    "min_mix_coef": 0.00,
                    "abs_tol_params": 1e-08,
                    "minimizer_options": {"maxiter": 100},
                },
            },
        }

    mc = EMClustering(**mc_params)
    mc.load_experiments(measurement_data)
    mc.run()
    mc.save_results()

if __name__ == "__main__":
    main()