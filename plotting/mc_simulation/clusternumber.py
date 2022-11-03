import numpy as np
import pandas as pd
import seaborn as sns

def get_score_correctly_identified_clusters(df_total, total_rank, init_routine):
    True_Cluster_number_perf = {}
    for True_Cluster_number in df_total.True_Cluster_number.unique():
        n_correct, n_total = 0, 0
        df1 = df_total[df_total.True_Cluster_number == True_Cluster_number]
        df1 = df1[df1.init_routine == init_routine]
        for dataset in df1.dataset.unique():
            df = df1[df1.dataset == dataset]
            df = df[df[total_rank] == 1]
            n_correct += df.identified_cluster.values[0]
            n_total += df.True_Cluster_number.values[0]
        True_Cluster_number_perf[True_Cluster_number] = [n_correct / n_total]
    df_routine_perf = pd.DataFrame.from_dict(data = True_Cluster_number_perf).melt(var_name="Initialization_Routine", value_name="correctly_identified_clusters")
    return df_routine_perf

def add_True_Cluster_number_performance_plot_identified(df_total, ax, settings,total_rank = "Total_prop_rank", init_routine="OPTICS"):
    df_score_performance =  get_score_correctly_identified_clusters(df_total, total_rank, init_routine)
    df_score_performance["Clusternumber"] = df_score_performance["Initialization_Routine"].astype(str)
    df_score_performance.sort_values("Clusternumber", inplace=True)
    sns.barplot(x=df_score_performance.correctly_identified_clusters, y=df_score_performance.Clusternumber, ax=ax)

    #ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

    ax.tick_params(axis='both', which='minor', labelsize=settings["axis_label_size"])
    ax.xaxis.label.set_size(settings["axis_label_size"])
    ax.yaxis.label.set_size(settings["axis_label_size"])
    ax.set( ylabel="True Clusternumber", xlim=(0,1))
    ax.set_title( label=f"{init_routine}\nCriterion: {total_rank}", color="w", size=settings["ax_title_size"])