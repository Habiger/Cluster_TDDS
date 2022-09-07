import pandas as pd
from model_selection.scores import scores


def create_scoreboard(results, X):
    for measure in scores.keys():
        results[measure] = []
    results["N_cluster"] = []

    for params in results["params"]:
        for criteria, items in scores.items():
            results[criteria].append(items["func"](X, params))
        results["N_cluster"].append(len(params) // 4)

    df = pd.DataFrame({key: val for key,val in results.items() if key != "params"})
    df_scores, df_scores_na = df.loc[~df["ll"].isna(), :], df.loc[df["ll"].isna(), :]
    #df_scores[[measure+"_rank" for measure in df_scores[scores.keys()]]] = df_scores[scores.keys()].rank().astype(int)
    for criteria, items in scores.items():
        df_scores.loc[:, criteria + "_rank"] = df_scores.loc[:, criteria].rank(method="min", **items["rank_params"]).astype(int)

    df_scores["TOTAL_score"] = df_scores[[criteria+"_rank" for criteria in scores.keys()]].sum(axis=1)
    df_scores["TOTAL_rank"] = df_scores["TOTAL_score"].rank(method="max").astype(int)

    df_scores = df_scores.sort_values("TOTAL_score").reset_index()
    df_scores.rename(columns={"index": "param_index"}, inplace=True)
    return df_scores, df_scores_na