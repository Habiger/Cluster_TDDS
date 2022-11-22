import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import seaborn as sns
import colorcet as cc
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mc_simulation.post_processing import get_prediction_df


def plot_clustering_assessment(model_data, df_scores, model_idx):
    sup_title_size = 40
    sup_axes_title_size = 30

    df = get_prediction_df(df_scores, model_data, model_idx)



    fig, ax = plt.subplots(facecolor='white')
    plt.box(False)



    N_pred_cluster = len(df.prediction_cluster.unique())
    n_row_grid = int(3+N_pred_cluster/5) +2
    fig.set_size_inches(n_row_grid*5, 20)
    gs = fig.add_gridspec(n_row_grid, 6)



    ########################################### general figure ###################################################################
    title = f"Analysis of Model No. {model_idx}"
    if title is not None:
        fig.suptitle(title, color="black", size=sup_title_size)
    ########################################### main plot: ax ######################################################################
    ax = fig.add_subplot(gs[:3, :6])
    ax.set_title("Classification of Points", size=sup_axes_title_size)
    # columns in df
    cluster_col= "cluster"
    cluster_pred_col = "prediction_cluster"

    alpha_correct, alpha_wrong = 1, 0.25
    df["alpha"] = np.where(df.cluster==df.identified_as_cluster, alpha_correct, alpha_wrong)

    avail_markers, avail_markers_means = ["$a$", "$b$", "$c$", "$d$", "$e$", "$f$", "$g$", "$h$", "$i$", "$j$"], ["$A$", "$B$", "$C$", "$D$", "$E$", "$F$", "$G$", "$H$", "$I$", "$J$"]
    marker_dict = {cl: avail_markers[i] for i, cl in enumerate(sorted(df.prediction_cluster.unique()))}
    marker_dict_means = {cl: avail_markers_means[i] for i, cl in enumerate(sorted(df.prediction_cluster.unique()))}
    df["marker"] = df["prediction_cluster"].apply(lambda x: marker_dict.get(x))

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
        ax.scatter(mean_x, mean_y, marker=marker_dict_means[pred_cl], c = "grey", s=300)

    ax.set_xlabel(r"$\log{\frac{\tau_e}{1\ s}}$", size=25, color="black")
    ax.set_ylabel(r"$\Delta V_{th}$", size=25, color="black")
    ax.tick_params(axis='both', which='major', labelsize=15, colors="black")

    ############################ legend ####################################################################################
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
    plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1,0.7), prop={"size": 16})
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ############################ responsibilities #########################################################################
    ax_subtitle = fig.add_subplot(gs[4, :5])
    ax_subtitle.set_title("Posterior Probabilities", size=sup_axes_title_size, pad=60)
    ax_subtitle._frameon = False
    cmap = sns.cubehelix_palette(as_cmap=True)
    cm_norm = mpl.colors.Normalize(vmin=0., vmax=1.0)
    params = model_data["inferred_mixtures"][model_idx]
    for i, cl_pred in enumerate(sorted(df.prediction_cluster.unique())):
        df_pred_cl = df[df.prediction_cluster == cl_pred]
        n_row = 4+i//5
        n_col = i%5
        ax_res = fig.add_subplot(gs[n_row, n_col])
        ax_res.scatter(np.log(df.x), df.y, c=df[f"gamma_{i}"], s=20, cmap=cmap, norm=cm_norm)
        ax_res.set_title(f"Predicted Cluster: {df_pred_cl.iloc[0, df_pred_cl.columns.get_loc('marker')]}", size=18)

        ax_res.scatter(np.log(params[1+i*4]), params[2+i*4], c="orange", label="Posterior Distribution Parameter")
        if i == 0:
            ax_res.legend(loc="center left", bbox_to_anchor=(0.1, 1.4), fontsize=20)

        #fig.colorbar()
    ax_cb = fig.add_subplot(gs[4:,5])
    plt.box(False)
    ax_cb2 = inset_axes(ax_cb, width="20%", height="80%", loc="center")
    ax_cb2.set_title("Probabilities", size = 20, color="black", pad=30)
    plt.box(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cm_norm)
    sm.set_array([])

    cb = fig.colorbar(sm, ticks=np.linspace(0,1, 11),cax=ax_cb2, orientation="vertical", shrink=0.5)

    fg_color="black"

    cb.ax.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color, labelsize=15)
    #cb.set_label("Probabilities", color="black", size=20, rotation=270, pad=20) #, labelpad=30

    ########################### original data #############################################################################
    ax_og = fig.add_subplot(gs[3, 4:])
    for cl in df.cluster.unique():
        df_cl = df[df.cluster == cl]
        ax_og.scatter(np.log(df_cl.x), df_cl.y, c=c_palette[cl], alpha=0.65)
    ax_og.set_title("True Cluster", size=22)
    plt.close()
    return fig