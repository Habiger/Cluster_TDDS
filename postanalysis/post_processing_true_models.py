import pandas as pd

from sklearn.metrics.cluster import adjusted_rand_score
from postanalysis.post_processing import predict_clusters


def include_true_clusternumber(df_results: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    for dataset in df_results.dataset.unique():
        idx = df_results.dataset == dataset
        df_results.loc[idx, "True_N_Cluster"] = len(model_data["datasets"][dataset].cluster.unique())
    df_results["True_N_Cluster"] = df_results["True_N_Cluster"].astype(int)
    return df_results


def include_prediction_quality_measures(df_scores: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """computes number of correctly predicted clusters and the adjusted rand score for each model"""
    row_idxs, numbers_of_identified_clusters, rand_scores = [], [], []
    for row_idx, row in df_scores.iterrows():
        df_pred = get_prediction_df(df_scores, model_data, row.model_idx)
        numbers_of_identified_clusters.append(_get_number_of_correctly_identified_clusters(df_pred))
        row_idxs.append(row_idx)
        rand_scores.append(adjusted_rand_score(df_pred["cluster"], df_pred["prediction_cluster"]))
    df_scores.loc[row_idxs, "number_identified_cluster"] = numbers_of_identified_clusters
    df_scores.loc[row_idxs, "adjusted_rand_score"] = rand_scores
    return df_scores


def get_prediction_df(df_scores: pd.DataFrame, model_data: dict, model_idx: int) -> pd.DataFrame:
    df = predict_clusters(df_scores, model_data, model_idx)
    cluster_maping_dict = _get_correctly_classified_mapping(df)
    df["identified_as_cluster"] = df.copy().replace({"prediction_cluster": cluster_maping_dict})["prediction_cluster"]
    return df


def _get_correctly_classified_mapping(df: pd.DataFrame) -> dict:
    df_count_cl_pair = df.groupby(["cluster", "prediction_cluster"]).count().reset_index()
    df_count_cl_pred = df.groupby(["prediction_cluster"]).agg({"x": "count"})
    df_count_cl_true = df.groupby(["cluster"]).agg({"x": "count"})

    mapping_dict = {cl_pred: "noise" for cl_pred in df_count_cl_pair.prediction_cluster.unique()}
    for (cl, cl_pred) in zip(df_count_cl_pair.cluster, df_count_cl_pair.prediction_cluster):
        condition_1, condition_2 = False, False
        number_of_points_correctly_pred = df_count_cl_pair.loc[
            (df_count_cl_pair.prediction_cluster == cl_pred) & (df_count_cl_pair.cluster == cl),
            "x",
        ].values[0]
        if (
            number_of_points_correctly_pred > 0.5 * df_count_cl_pred.loc[cl_pred, "x"]
        ):  # condition 1: majority of points in pred_cluster are in true_cluster
            condition_1 = True
        if (
            number_of_points_correctly_pred > 0.5 * df_count_cl_true.loc[cl, "x"]
        ):  # condition 2: majority of points of true_cluster are in pred_cluster
            condition_2 = True
        if condition_1 and condition_2:
            mapping_dict[cl_pred] = cl
    return mapping_dict


def _get_number_of_correctly_identified_clusters(df_pred: pd.DataFrame) -> int:
    correctly_identified_clusters = 0
    for cl in df_pred.cluster.unique():
        df_temp = df_pred[df_pred.cluster == cl]
        count_correctly_identified_points = sum(df_temp.cluster == df_temp.identified_as_cluster)
        number_of_points = len(df_temp.index)
        if count_correctly_identified_points > number_of_points / 2:
            correctly_identified_clusters += 1
    return correctly_identified_clusters
