import pandas as pd
from clustering.em.em_algorithm import gamma


def predict_clusters(df_scores: pd.DataFrame, model_data: dict, model_idx: int):
    """includes responsibilities for predicted distributions into dataset df

    Args:
        model_data (dict): contains the predicted parameters of the underlying mixture and the dataset
        df_scores (pd.DataFrame): contains the scores of the different competing models
        model_idx (int): model to use for prediction

    Returns:
        df: responsibilities for clusters included
    """
    dataset_idx = df_scores[df_scores.model_idx == model_idx].dataset.values[0]
    df = model_data["datasets"][int(dataset_idx)].copy()
    X = df[["x", "y"]].to_numpy(copy=True)
    inferred_mixture_params = model_data["inferred_mixtures"][model_idx]

    K = len(inferred_mixture_params) // 4
    for cl in range(K):
        df[f"gamma_{cl}"] = gamma(X, inferred_mixture_params, cl)
    df["prediction_cluster"] = df[[col for col in df.columns if "gamma" in col]].idxmax(axis=1).str[-1]
    return df
