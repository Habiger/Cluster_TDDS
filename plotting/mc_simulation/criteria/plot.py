import matplotlib.pyplot as plt
from plotting.mc_simulation.criteria.processing import wrapper_get_correctly_identified_clusters, select_init_routine


def plot_comparison_criteria(df_total, settings):
    init_routines, ranks = df_total.init_routine.unique(), [col for col in df_total.columns if "rank" in col]
    df_all = wrapper_get_correctly_identified_clusters(df_total, init_routines, ranks)

    fig, axs = plt.subplots(nrows=len(init_routines))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    col_order = ['ll', 'AIC', 'BIC',  'MML','CH', 'silhouette','TOTAL', 'Total_prop', 'Best_Model']
    for i, init_routine in enumerate(init_routines):
        ax = axs[i]
        df = select_init_routine(df_all, init_routine)[col_order]
        df.plot(kind="bar", ax=ax)
        axs[i].set_title(f"Initialization Routine: {init_routine}", color ="white")
        axs[i].set_ylim((0,1))
        axs[i].xaxis.set_tick_params(rotation=0)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # Put a legend to the right of the current axis
        ax.legend(title="Model selected by\n",loc='center left', bbox_to_anchor=(1, 0.6))
        
    fig.suptitle("Percentage of correctly identified Clusters\nby different criteria", color="w", size = settings["sup_title_size"])
    fig.set_size_inches(15,10)
    plt.close()
    return fig, df_all

