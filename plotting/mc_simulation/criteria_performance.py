import numpy as np
import pandas as pd
import seaborn as sns

from model_selection.scores import scores

def get_score_performance(df_total):
    criteria = list(scores.keys())
    rank_criteria = [crit+"_rank" for crit in criteria]  + ["Total_prop_rank"]

    correct_pred = {}
    for rank_crit in rank_criteria:
        n_correct = 0
        n_total = 0
        for dataset in df_total.dataset.unique():
            df = df_total[df_total.dataset == dataset]
            df = df[df[rank_crit] == 1]   # top rank
            df = df[df.ll == np.max(df.ll)] # if multiple top ranks, best ll
            if df.N_cluster.values[0] == df.True_Cluster_number.values[0]:
                n_correct += 1
            n_total += 1
            
        correct_pred[rank_crit] = [n_correct/n_total]
    df_score_performance = pd.DataFrame.from_dict(data=correct_pred)
    df_score_performance = df_score_performance.melt(var_name="Criterion", value_name="success_rate")
    return df_score_performance


def add_score_performance_plot(df_total, ax, settings):
    df_score_performance = get_score_performance(df_total)
    sns.barplot(y=df_score_performance.Criterion, x=df_score_performance.success_rate, ax=ax)

    #ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

    ax.tick_params(axis='both', which='minor', labelsize=settings["axis_label_size"])
    ax.xaxis.label.set_size(settings["axis_label_size"])
    ax.yaxis.label.set_size(settings["axis_label_size"])
    ax.set( ylabel="", xlim=(0,1))
    ax.set_title( label="Model Selection Criteria Performance", color="w", size=settings["ax_title_size"])