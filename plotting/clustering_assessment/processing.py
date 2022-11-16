from post_processing.identfy_clusters import get_prediction_df


def select_prediction_df(run_data, run_results, init_routine, dataset, criterion, param_idx=None):
    df_exp = run_data[dataset].df
    results = {"params":run_results[init_routine]["em_results"][dataset]}
    df_scores = run_results[init_routine]["df_scores"][dataset].sort_values(criterion) #must be ranked according to selected criterion
    df_pred = get_prediction_df(df_exp, results, df_scores, param_idx=param_idx)
    return df_pred

def select_prediction_df_single(results, df_scores, df_exp, criterion, param_idx=None):
    df_scores = df_scores.sort_values(criterion)
    return get_prediction_df(df_exp, results, df_scores, param_idx=param_idx)