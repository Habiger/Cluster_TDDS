import pandas as pd
#from post_processing.identfy_clusters import get_prediction_df
from tqdm import tqdm

import pandas as pd
import numpy as np

from em_algorithm.em_funcs import gamma


def get_prediction_df(df_scores, model_data, param_idx):
    df_with_responsibilities = predict_clusters(df_scores, model_data, param_idx)
    #df_pred_params = calculate_cluster_params(df_with_responsibilities)
    #df_sample_params = calculate_cluster_params_from_drawn_samples(df_with_responsibilities) # not needed here too
    #df = df_sample_params.merge(df_pred_params, how="cross")
    #df["center_distance"] = calc_dist(df.x_mean_og, df.y_mean_og, df.x_mean, df.y_mean) #BUG not needed here?!
    #df = df.sort_values("center_distance").groupby("cluster", as_index=False).first()
    df = df.merge(df_with_responsibilities)
    cluster_maping_dict = get_correctly_classified_mapping(df)
    df["identified_as_cluster"] = df.copy().replace({"prediction_cluster": cluster_maping_dict})["prediction_cluster"]
    return df

def predict_clusters(model_data, df_scores, model_idx):
    """includes responsibilities for predicted parameters of distributions into df

    Args:
        df (pd.DataFrame): original experiment data
        model_data (dict): contains the predicted parameters of the underlying mixture
        df_scores (pd.DataFrame): contains the scores of the different competing models

    Returns:
        df: responsibilities for clusters included
    """
    dataset_idx = df_scores[df_scores.model_idx == model_idx].dataset.values[0]
    df = model_data["datasets"][dataset_idx].copy()
    X = df[["x", "y"]].to_numpy(copy=True)
    inferred_mixture_params = model_data["inferred_mixtures"][model_idx]
    
    K = len(inferred_mixture_params) //4
    for cl in range(K):
        df[f"gamma_{cl}"]  = gamma(X, inferred_mixture_params, cl)
    df["prediction_cluster"] =df[[col for col in df.columns if "gamma" in col]].idxmax(axis=1).str[-1]
    return df

def get_correctly_classified_mapping(df2):
    df_count_cl_pair = df2.groupby(["cluster", "prediction_cluster"]).count().reset_index()
    df_count_cl_pred = df2.groupby(["prediction_cluster"]).agg({"x": "count"})
    df_count_cl_true = df2.groupby(["cluster"]).agg({"x": "count"})

    mapping_dict = {cl_pred: "noise" for cl_pred in df_count_cl_pair.prediction_cluster.unique()}
    for (cl, cl_pred) in zip(df_count_cl_pair.cluster, df_count_cl_pair.prediction_cluster):
        condition_1, condition_2 = False, False
        number_of_points_correctly_pred = df_count_cl_pair.loc[(df_count_cl_pair.prediction_cluster == cl_pred)&(df_count_cl_pair.cluster == cl), "x"].values[0]
        if number_of_points_correctly_pred > 0.5*df_count_cl_pred.loc[cl_pred, "x"]: # majority of points in pred_cluster are member of cluster
            condition_1 = True
        if number_of_points_correctly_pred > 0.5*df_count_cl_true.loc[cl, "x"]:
            condition_2 = True
        if condition_1 and condition_2:
            mapping_dict[cl_pred] = cl
    return mapping_dict


    
def get_number_of_correctly_identified_clusters(df_pred):
    correctly_identified_clusters = 0
    for cl in df_pred.cluster.unique():
        df_temp = df_pred[df_pred.cluster == cl]
        count_correctly_identified_points = sum(df_temp.cluster == df_temp.identified_as_cluster)
        number_of_points = len(df_temp.index)
        if count_correctly_identified_points > number_of_points/2:
            correctly_identified_clusters += 1
    return correctly_identified_clusters

def post_analysis(df_scores, model_data):
    """identifies number of correctly predicted clusters
    """
    row_idxs, numbers_of_identified_clusters = []
    for row_idx, row in df_scores.iterrows():
        df_pred = get_prediction_df(df_scores, model_data, row.model_idx)
        numbers_of_identified_clusters.append(get_number_of_correctly_identified_clusters(df_pred))
        row_idxs.append(row_idx)
    df_scores.loc[row_idxs, "number_identified_cluster"] = numbers_of_identified_clusters
    return df_scores

"""
    df_list = []
    for init in df_scores.keys():
        print(init)
        for i, df in tqdm(enumerate(df_scores[init]["df_scores"])):
            df["init_routine"] = init
            df["dataset"] = i
            df["True_Cluster_number"] = model_data[i].n_cluster
            for j in df.index:
                df_pred = get_prediction_df(
                    model_data[i].df, 
                    {"params": df_scores[init]["em_results"][i]},
                    df[df.index==j]
                ) 
                df.loc[df.index==j, "identified_cluster"] = get_number_of_correctly_identified_clusters(df_pred)
            df_list.append(df)
    df_total = pd.concat(df_list)
    return df_total"""