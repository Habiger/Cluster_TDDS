import matplotlib.pyplot as plt


def plot_init_routine_comparison_best_models(data):
    """uses `data` from `plot_comparison_criteria` in folder `criteria`
    """
    df = data[data["criteria"] == "Best_Model_"]
    df = df.pivot_table(values="correctly_identified_clusters", columns="init_routine", index ="True cluster number")
    df.plot(kind="bar")
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(title = "init_routine", loc='center left', bbox_to_anchor=(1, 0.6), fontsize=20, title_fontsize=20)
    ax.set_title("Percentage of correctly identified clusters by best model", color="w", size=30)
    ax.xaxis.set_tick_params(rotation=0, labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.label.set_size(15)

    fig = plt.gcf()
    fig.set_size_inches(15,10)
    plt.close()
    return fig