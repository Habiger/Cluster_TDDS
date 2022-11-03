import numpy as np
import pandas as pd
import seaborn as sns


def get_init_rout_performance(df_total, total_rank):
    init_rout_perf = {}
    for init_routine in df_total.init_routine.unique():
        n_correct, n_total = 0, 0
        for dataset in df_total.dataset.unique():
            df = df_total[df_total.init_routine == init_routine]
            df = df[df.dataset == dataset]
            df = df[df[total_rank] == 1]
            if df.N_cluster.values[0] == df.True_Cluster_number.values[0]:
                n_correct += 1
            n_total += 1
            init_rout_perf[init_routine] = [n_correct / n_total]
    df_routine_perf = pd.DataFrame.from_dict(data = init_rout_perf).melt(var_name="Initialization_Routine", value_name="correct_number_of_clusters")
    return df_routine_perf


def get_score_correctly_identified_clusters(df_total, total_rank):
    init_rout_perf = {}
    for init_routine in df_total.init_routine.unique():
        n_correct, n_total = 0, 0
        for dataset in df_total.dataset.unique():
            df = df_total[df_total.init_routine == init_routine]
            df = df[df.dataset == dataset]
            df = df[df[total_rank] == 1]
            n_correct += df.identified_cluster.values[0]
            n_total += df.True_Cluster_number.values[0]
            init_rout_perf[init_routine] = [n_correct / n_total]
    df_routine_perf = pd.DataFrame.from_dict(data = init_rout_perf).melt(var_name="Initialization_Routine", value_name="correctly_identified_clusters")
    return df_routine_perf


def add_init_rout_performance_plot(df_total, ax, settings,total_rank = "Total_prop_rank"):
    df_score_performance =  get_init_rout_performance(df_total, total_rank)
    sns.barplot(y=df_score_performance.Initialization_Routine, x=df_score_performance.correct_number_of_clusters, ax=ax)

    #ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

    ax.tick_params(axis='both', which='minor', labelsize=settings["axis_label_size"])
    ax.xaxis.label.set_size(settings["axis_label_size"])
    ax.yaxis.label.set_size(settings["axis_label_size"])
    ax.set( ylabel="", xlim=(0,1))
    ax.set_title( label=f"Initialization Routine Performance\nCriterion: {total_rank}", color="w", size=settings["ax_title_size"])

def add_init_rout_performance_plot_identified(df_total, ax, settings,total_rank = "Total_prop_rank"):
    df_score_performance =  get_score_correctly_identified_clusters(df_total, total_rank)
    sns.barplot(y=df_score_performance.Initialization_Routine, x=df_score_performance.correctly_identified_clusters, ax=ax)

    #ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

    ax.tick_params(axis='both', which='minor', labelsize=settings["axis_label_size"])
    ax.xaxis.label.set_size(settings["axis_label_size"])
    ax.yaxis.label.set_size(settings["axis_label_size"])
    ax.set( ylabel="", xlim=(0,1))
    ax.set_title( label=f"Initialization Routine Performance\nCriterion: {total_rank}", color="w", size=settings["ax_title_size"])