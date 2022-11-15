import matplotlib.pyplot as plt
from plotting.mc_simulation.best_model_position.processing import get_position_of_best_models

def plot_ranks_of_best_model(df_total, rank, N_experiments, N_runs_per_clusternumber, init_routine=False):
    if init_routine:
        df_total = df_total[df_total.init_routine==init_routine]
    else:
        df_total = df_total
    df_position, df_rank_cluster, df_position_inside_clusternumber = get_position_of_best_models(df_total)
    n = len(df_position.index)
    fig = plt.figure()
    fig.set_size_inches(15,10)
    gs = fig.add_gridspec(2,2)
    fig.suptitle(f"Rank of Best Model\naccording to Criterion {rank}", color="white")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.hist(df_rank_cluster[rank], bins=range(N_runs_per_clusternumber+1), align="left");
    ax1.set_xticks(range(N_runs_per_clusternumber))
    ax1.set_title("Inter-clusternumber rank of best model's clusternumber", color="white")

    ax2.hist(df_position_inside_clusternumber[rank], bins=range(N_runs_per_clusternumber+1), align="left")
    ax2.set_xticks(range(N_runs_per_clusternumber))
    ax2.set_title("Intra-clusternumber rank of best model", color="white")

    ax3.hist(df_position[rank], bins=N_experiments)
    ax3.set_title("Overall rank of best model", color="white")
    #ax3.set_xticks(range(N_experiments))
    plt.close()
    return fig