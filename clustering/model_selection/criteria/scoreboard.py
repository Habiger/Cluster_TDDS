import numpy as np
import pandas as pd

from clustering.model_selection.criteria.functions import criteria_dict

def create_scoreboard(df_results: pd.DataFrame, model_data: dict, sort_by: list = ["dataset","Total_rank_rank", "ll_score"]):
    """Calculates and includes a variety of scores for each model into df_results. Divergent/Singular models will be excluded.

    Args:
        df_results (pd.DataFrame): every row corresponds to a single model
        model_data (dict): contains experiment datasets and the inferred mixture parameters from the models
        sort_by (list, optional): sorts the rows in the returned dataframe by the given features. Defaults to ["dataset","Total_rank_rank", "ll_score"].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: df_results with scores | df_results for divergent models
    """
    df_scores = _calculate_scores(df_results, model_data)
    df_scores, df_scores_singular = _filter_singular_models(df_scores)
    df_scores = _calculate_total_proportional_score(df_scores, criteria_dict)
    df_scores = _include_score_ranks(df_scores, criteria_dict)
    df_scores = df_scores.sort_values(sort_by)
    return df_scores, df_scores_singular


def _calculate_scores(df_results: pd.DataFrame, model_data: dict):
    calculated_scores = {}
    for criterion in criteria_dict.keys():
        calculated_scores[criterion] = []
        for model in df_results.itertuples():
            X = model_data["datasets"][model.dataset][["x", "y"]].to_numpy() 
            inferred_mixture = model_data["inferred_mixtures"][model.model_idx]
            score = criteria_dict[criterion]["func"](X, inferred_mixture)
            calculated_scores[criterion].append(score)

    for criterion, score in calculated_scores.items():
        df_results[f"{criterion}_score"] = score
    return df_results

def _filter_singular_models(df_scores: pd.DataFrame):
    df_scores_wo_na, df_scores_with_na = df_scores.loc[~df_scores["ll_score"].isna(), :].copy(), df_scores.loc[df_scores["ll_score"].isna(), :].copy()
    return df_scores_wo_na, df_scores_with_na

def _calculate_total_proportional_score(df_scores: pd.DataFrame, criteria_dict: dict):
    criteria_dict = {criterion: items for criterion, items in criteria_dict.items() if items["use_in_total_score"]}
    for criterion, items in criteria_dict.items():
        if items["rank_params"]["ascending"]:
            score_order_corr = -df_scores[criterion + "_score"]
        else:
            score_order_corr = df_scores[criterion + "_score"]
        score_shifted = score_order_corr - np.min(score_order_corr)
        score_scaled = score_shifted / max(score_shifted)
        df_scores[criterion+"_relative"] = score_scaled

    df_scores["Total_proportional_score"] = df_scores[[criterion + "_relative" for criterion in criteria_dict.keys()]].sum(axis=1)
    df_scores["Total_proportional_score"] = df_scores["Total_proportional_score"] / len(criteria_dict.keys())
    df_scores.drop(columns=[criterion+"_relative" for criterion in criteria_dict.keys()], inplace=True)
    return df_scores


def _include_score_ranks(df_scores: pd.DataFrame, criteria_dict: dict):
    for dataset in df_scores.dataset.unique():
        idxs = df_scores.dataset == dataset
        for criteria, items in criteria_dict.items():
            df_scores.loc[idxs, criteria + "_rank"] = df_scores.loc[idxs, criteria+"_score"].rank(method="dense", **items["rank_params"])
        # total proportional score:
        df_scores.loc[idxs, "Total_proportional" + "_rank"] = df_scores.loc[idxs, "Total_proportional_score"].rank(method="dense", ascending=False)
        # total rank score
        df_scores.loc[idxs, "Total_rank_score"] = df_scores.loc[idxs, [criteria+"_rank" for criteria, item in criteria_dict.items() if item["use_in_total_score"]]].sum(axis=1)
        df_scores.loc[idxs, "Total_rank_rank"] = df_scores.loc[idxs, "Total_rank_score"].rank(method="dense").astype(int)
    rank_cols = [col for col in df_scores.columns if "_rank" in col]
    df_scores[rank_cols] = df_scores[rank_cols].astype(int)
    return df_scores



