import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

import colorcet as cc

import numpy as np

from plotting.clustering_assessment.processing import select_prediction_df, select_prediction_df_single

def plot_clustering_assessment(run_data, run_results, init_routine, dataset, criterion, param_idx=None):
    df_pred = select_prediction_df(run_data, run_results, init_routine, dataset, criterion, param_idx=param_idx)
    fig = _plot_clustering_assessment(df_pred)
    return fig


def plot_clustering_assessment_single(results, df_scores, df_exp, criterion, param_idx=None):
    df_pred = select_prediction_df_single(results, df_scores, df_exp, criterion, param_idx=None)
    fig = _plot_clustering_assessment(df_pred)
    return fig

def _plot_clustering_assessment(df, title ="Assessment of Clustering"):
    
    # columns in df
    cluster_col= "cluster"
    cluster_pred_col = "prediction_cluster"

    alpha_correct, alpha_wrong = 1, 0.25
    df["alpha"] = np.where(df.cluster==df.identified_as_cluster, alpha_correct, alpha_wrong)

    avail_markers, avail_markers_means = ["$a$", "$b$", "$c$", "$d$", "$e$", "$f$", "$g$", "$h$", "$i$", "$j$"], ["$A$", "$B$", "$C$", "$D$", "$E$", "$F$", "$G$", "$H$", "$I$", "$J$"]

    marker_dict = {cl: avail_markers[i] for i, cl in enumerate(sorted(df.prediction_cluster.unique()))}
    marker_dict_means = {cl: avail_markers_means[i] for i, cl in enumerate(sorted(df.prediction_cluster.unique()))}
    df["marker"] = df["prediction_cluster"].apply(lambda x: marker_dict.get(x))


    fig, ax = plt.subplots()
    fig.set_size_inches(25,15)
    c_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    clusters = sorted(df[cluster_col].unique())
    predicted_clusters = sorted(df.prediction_cluster.unique())
    for i, cl in enumerate(clusters):
        df_subset =  df[df[cluster_col] == cl]
        mean_x, mean_y = np.log(np.mean(df_subset.x)), np.mean(df_subset.y)
        ax.scatter(mean_x, mean_y, marker="x", c = c_palette[cl], s=300, label= f"cluster {cl}")
        for pred_cl in sorted(df_subset.prediction_cluster.unique()):
            df_subset2 = df_subset[df_subset[cluster_pred_col]  ==pred_cl]
            ax.scatter(np.log(df_subset2.x), df_subset2.y, c=c_palette[cl], alpha=df_subset2.alpha, marker=marker_dict[pred_cl], s=150)    
    for pred_cl in sorted(df.prediction_cluster.unique()):
        df_subset = df[df[cluster_pred_col] == pred_cl]
        mean_x, mean_y = np.log(np.mean(df_subset.x)), np.mean(df_subset.y)
        ax.scatter(mean_x, mean_y, marker=marker_dict_means[pred_cl], c = "grey", s=200)

    ax.set_xlabel(r"$\log{\frac{\tau_e}{1\ s}}$", size=25)
    ax.set_ylabel(r"$\Delta V_{th}$", size=25)
    ax.tick_params(axis='both', which='major', labelsize=15)



    # True clusters
    handles = [mlines.Line2D([0],[0],label=r"$\bf{True\ Clusters}$", color="w", ls="")]
    handles += [mlines.Line2D([0],[0], color=c_palette[cl], label=f"Cluster {cl}", marker="o", ls="") for cl in clusters]
    ## means
    handles += [mlines.Line2D([0],[0],label=r" ", color="w", ls="")] + [mlines.Line2D([0],[0],label=r"Empirical Means", color="black", ls="", marker="x", markersize=10)]
    # Predicted clusters
    handles += [mlines.Line2D([0],[0],label=r" ", color="w", ls="")] + [mlines.Line2D([0],[0],label=r"$\bf{Predicted\ Clusters}$", color="w", ls="")]
    handles +=  [mlines.Line2D([0],[0],label=f"Cluster {marker_dict[pred_cl]}", color="black", ls="", marker=marker_dict[pred_cl], markersize=10) for pred_cl in predicted_clusters]
    ## means
    handles += [mlines.Line2D([0],[0],label=r" ", color="w", ls="")] + [mlines.Line2D([0],[0],label=r"Empricial Means", color="black", ls="", marker=r"$A,B,..$", markersize=30)]
    #Classification
    handles += [mlines.Line2D([0],[0],label=r" ", color="w", ls="")] + [mlines.Line2D([0],[0],label=r"$\bf{Classification\ of\ Points}$", color="w", ls="")]
    handles += [mlines.Line2D([0],[0], color="black", label=f"correctly classified", alpha=alpha_correct, marker="o", ls="")]
    handles += [mlines.Line2D([0],[0], color="black", label=f"misclassified", alpha=alpha_wrong, marker="o", ls="")]
    #leg_true_means = ax.legend(title="True empirical Means", loc="best")
    plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1,0.6))
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])


    #plt.legend(handles=color_patches)

    if title is not None:
        fig.suptitle(title, color="white", size=30)
    plt.close()
    return fig