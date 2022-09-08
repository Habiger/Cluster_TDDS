import pandas as pd
import numpy as np
from model_selection.scores import scores


def create_scoreboard(results, X):
    for measure in scores.keys():
        results[measure] = []
    results["N_cluster"] = []

    for params in results["params"]:  #BUG setting copy warnings
        for criteria, items in scores.items():
            results[criteria].append(items["func"](X, params))
        results["N_cluster"].append(len(params) // 4)

    df = pd.DataFrame({key: val for key,val in results.items() if key != "params"})
    df_scores, df_scores_na = df.loc[~df["ll"].isna(), :], df.loc[df["ll"].isna(), :]
    #df_scores[[measure+"_rank" for measure in df_scores[scores.keys()]]] = df_scores[scores.keys()].rank().astype(int)
    for criteria, items in scores.items():
        df_scores.loc[:, criteria + "_rank"] = df_scores.loc[:, criteria].rank(method="min", **items["rank_params"]).astype(int).copy()

    df_scores["TOTAL_score"] = df_scores[[criteria+"_rank" for criteria in scores.keys()]].sum(axis=1).copy()
    df_scores["TOTAL_rank"] = df_scores["TOTAL_score"].rank(method="min").astype(int).copy()


    df_scores = calculate_total_proportional_rank(df_scores, scores)
    df_scores = df_scores.sort_values("Total_prop_rank", ascending=False).reset_index()
    df_scores.rename(columns={"index": "param_index"}, inplace=True)
    return df_scores, df_scores_na


def calculate_total_proportional_rank(df_scores, scores):
    for score, items in scores.items():
        if items["rank_params"]["ascending"]:
            score_order_corr = -df_scores[score]
        else:
            score_order_corr = df_scores[score]
        score_shifted = score_order_corr - np.min(score_order_corr)
        score_scaled = score_shifted / max(score_shifted)
        df_scores[score+"_relative"] = score_scaled

    df_scores["Total_score_prop"] = df_scores[[score + "_relative" for score in scores.keys()]].sum(axis=1).copy()
    df_scores["Total_score_prop"] = df_scores["Total_score_prop"] / len(scores.keys())
    df_scores.drop(columns=[score+"_relative" for score in scores.keys()], inplace=True)
    df_scores["Total_prop_rank"] = df_scores["Total_score_prop"].rank(method="min", ascending=False).astype(int)
    return df_scores