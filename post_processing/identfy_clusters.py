import pandas as pd
import numpy as np

from em_algorithm.em_funcs import gamma


def get_prediction_df(df_exp, model_data, df_scores, param_idx=None):
    df_with_responsibilities = predict_clusters(df_exp, model_data, df_scores, param_idx)
    #df_pred_params = calculate_cluster_params(df_with_responsibilities)
    #df_sample_params = calculate_cluster_params_from_drawn_samples(df_with_responsibilities) # not needed here too
    #df = df_sample_params.merge(df_pred_params, how="cross")
    #df["center_distance"] = calc_dist(df.x_mean_og, df.y_mean_og, df.x_mean, df.y_mean) #BUG not needed here?!
    df = df.sort_values("center_distance").groupby("cluster", as_index=False).first()
    df = df.merge(df_with_responsibilities)
    cluster_maping_dict = get_correctly_classified_mapping(df)
    df["identified_as_cluster"] = df.copy().replace({"prediction_cluster": cluster_maping_dict})["prediction_cluster"]
    return df


def gamma_cols(df):
    return [col for col in df.columns if "gamma" in col]

def predict_clusters(df_experiment, model_data, df_scores, model_idx=None):
    """includes responsibilities for predicted parameters of distributions into df

    Args:
        df (_type_): original experiment data
        results (_type_): contains the predicted parameters of the underlying mixture
        df_scores (_type_): contains the scores of the different competing models

    Returns:
        df: responsibilities for clusters included
    """
    df = df_experiment.copy()
    X = df[["x", "y", "cluster"]].to_numpy()
    #best_model_idx = 0
    if model_idx is None:       #TODO only do one thing
        inferred_mixture_params = model_data["inferred_mixtures"][df_scores.model_idx.iloc[0]]
    else:
        inferred_mixture_params = model_data["inferred_mixtures"][model_idx]
    
    K = len(inferred_mixture_params) //4
    for cl in range(K):
        df[f"gamma_{cl}"]  = gamma(X, inferred_mixture_params, cl)
    df["prediction_cluster"] =df[gamma_cols(df)].idxmax(axis=1).str[-1]
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


########################################################################################################################

def calculate_cluster_params(df):  #BUG probably not needed
    df_cl_grouped = df.groupby(["prediction_cluster"]).agg({"x": ["mean"], "y": ["mean", "std"]})
    df_cl_grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_cl_grouped.columns.values]
    df_cl_grouped = df_cl_grouped.reset_index()  
    return df_cl_grouped

def calc_dist(x1, y1, x2, y2): #BUG probably not needed
    return np.sqrt((np.log(x1)-np.log(x2))**2 + (y1-y2)**2)

def calculate_cluster_params_from_drawn_samples(df): #BUG probably not needed
    """params from the sampled data points

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_cl_grouped = df.groupby(["cluster"]).agg({"x": ["mean"], "y": ["mean", "std"]})
    df_cl_grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_cl_grouped.columns.values]
    df_cl_grouped.columns = [col+"_og" for col in df_cl_grouped.columns]
    df_cl_grouped = df_cl_grouped.reset_index()  
    return df_cl_grouped

def get_original_cluster_params_df(exp): #BUG probably not needed
    """params from which there was sampled

    Args:
        exp (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_dict = {
        "true_cluster": [],
        "x_loc": [],
        "y_loc": []
    }

    for i, cluster in enumerate(exp.cluster):
        df_dict["true_cluster"].append(i)
        df_dict["x_loc"].append(cluster.param_exp["scale"])
        df_dict["y_loc"].append(cluster.param_norm["loc"])
    return pd.DataFrame.from_dict(data=df_dict)