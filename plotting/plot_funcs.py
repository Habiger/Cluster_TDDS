import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc

from em_algorithm.em_funcs import gamma

def plot_responsibilities(experiment, params_array):
    df = experiment.df.copy()
    K = len(params_array)//4
    for cl in range(K):
        df[f"gamma_{cl}"]  = gamma(experiment.X, params_array, cl)
    cmap = sns.cubehelix_palette(as_cmap=True)
    ncols, nrows = 5, max(2, K//5 + 1)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    fig.set_size_inches(20,10)
    for i in range(K):
        j = i//5
        k = i%5
        points = ax[j, k].scatter(np.log(df.x), df.y, c=df[f"gamma_{i}"], s=20, cmap=cmap, )
        #ax[j, k].scatter(np.log(experiment.cluster[i].param_exp["scale"]), experiment.cluster[i].param_norm["loc"], c="red")
        #ax[j, k].scatter(np.log(experiment.cluster[j].param_exp["scale"]), experiment.cluster[j].param_norm["loc"], c="blue")
        ax[j, k].scatter(np.log(params_array[1+i*4]), params_array[2+i*4], c="orange")
        ax[j, k].set_title(f"Cluster {i}", size=15, color="white")
    fig.suptitle("Reponsibilites $\gamma$", color="white", size = 30)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    fig.colorbar(points, cax=cbar_ax)
    plt.close()
    return fig


def plot_cluster(df, cluster_col, title=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    c_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    for i, cl in enumerate(sorted(df[cluster_col].unique())):
        df_subset =  df[df[cluster_col] == cl]
        if cl==-1:
            ax.scatter(np.log(df_subset.x), df_subset.y, label="Noise", c="black", alpha=0.2)
        else:
            ax.scatter(np.log(df_subset.x), df_subset.y, label=cl, c=c_palette[cl], alpha=0.8)    
            mean_x, mean_y = np.log(np.mean(df_subset.x)), np.mean(df_subset.y)
            ax.scatter(mean_x, mean_y, marker="x", c = c_palette[cl], s=200)
    ax.set_xlabel(r"$\log{\frac{\tau_e}{1\ s}}$", size=25)
    ax.set_ylabel(r"$\Delta V_{th}$", size=25)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(title="Cluster")
    if title is not None:
        fig.suptitle(title, color="white", size=30)
    plt.close()
    return fig