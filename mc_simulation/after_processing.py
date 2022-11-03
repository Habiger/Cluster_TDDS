import pandas as pd
from post_processing.identfy_clusters import get_prediction_df
from tqdm import tqdm

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



    
def get_number_of_correctly_identified_clusters(df_pred):
    correctly_identified_clusters = 0
    for cl in df_pred.cluster.unique():
        df_temp = df_pred[df_pred.cluster == cl]
        count_correctly_identified_points = sum(df_temp.cluster == df_temp.identified_as_cluster)
        number_of_points = len(df_temp.index)
        if count_correctly_identified_points > number_of_points/2:
            correctly_identified_clusters += 1
    return correctly_identified_clusters

def process_run_results2(run_results, run_data):

    df_list = []
    for init in run_results.keys():
        print(init)
        for i, df in tqdm(enumerate(run_results[init]["df_scores"])):
            df["init_routine"] = init
            df["dataset"] = i
            df["True_Cluster_number"] = run_data[i].n_cluster
            for j in df.index:
                df_pred = get_prediction_df(
                    run_data[i].df, 
                    {"params": run_results[init]["em_results"][i]},
                    df[df.index==j]
                ) 
                df.loc[df.index==j, "identified_cluster"] = get_number_of_correctly_identified_clusters(df_pred)
            df_list.append(df)
    df_total = pd.concat(df_list)
    return df_total