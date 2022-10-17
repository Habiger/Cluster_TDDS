import pandas as pd

def process_run_results(run_results, run_data):
    df_list = []
    for init in run_results.keys():
        for i, df in enumerate(run_results[init]["df_scores"]):
            df["init_routine"] = init
            df["dataset"] = i
            df["True_Cluster_number"] = run_data[i].n_cluster
            df_list.append(df)
    df_total = pd.concat(df_list)
    return df_total