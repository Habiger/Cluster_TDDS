import pandas as pd

def get_score_correctly_identified_clusters(df_total, criterion, init_routine):
    True_Cluster_number_perf = {}
    n_correct_all_clusternumbers, n_total_all_clusternumbers = 0, 0
    for True_Cluster_number in df_total.True_Cluster_number.unique():
        n_correct, n_total = 0, 0
        df1 = df_total[df_total.True_Cluster_number == True_Cluster_number]
        df1 = df1[df1.init_routine == init_routine]
        for dataset in df1.dataset.unique():
            df = df1[df1.dataset == dataset]
            df = df[df[criterion] == 1]
            n_correct += df.identified_cluster.values[0]
            n_total += df.True_Cluster_number.values[0]
        True_Cluster_number_perf[True_Cluster_number] = [n_correct / n_total]
        n_correct_all_clusternumbers += n_correct
        n_total_all_clusternumbers += n_total
    True_Cluster_number_perf["All_clusternumbers"] = [n_correct_all_clusternumbers / n_total_all_clusternumbers]
    df_routine_perf = pd.DataFrame.from_dict(data = True_Cluster_number_perf).melt(var_name="True cluster number", value_name="correctly_identified_clusters")
    return df_routine_perf

def get_correctly_identified_clusters_by_best_model(df_total, init_routine):
    True_Cluster_number_perf = {}
    n_correct_all_clusternumbers, n_total_all_clusternumbers = 0, 0
    for True_Cluster_number in df_total.True_Cluster_number.unique():
        n_correct, n_total = 0, 0
        df1 = df_total[df_total.True_Cluster_number == True_Cluster_number]
        df1 = df1[df1.init_routine == init_routine]
        for dataset in df1.dataset.unique():
            df = df1[df1.dataset == dataset]
            n_correct += df.identified_cluster.max()
            n_total += df.True_Cluster_number.values[0]
        True_Cluster_number_perf[True_Cluster_number] = [n_correct / n_total]
        n_correct_all_clusternumbers += n_correct
        n_total_all_clusternumbers += n_total
    True_Cluster_number_perf["All_clusternumbers"] = [n_correct_all_clusternumbers / n_total_all_clusternumbers]
    df_routine_perf = pd.DataFrame.from_dict(data = True_Cluster_number_perf).melt(var_name="True cluster number", value_name="correctly_identified_clusters")
    return df_routine_perf

def wrapper_get_correctly_identified_clusters(df_total, init_routines, ranks):
    df_list = []
    for init_routine in init_routines:
        for rank in ranks:
            df = get_score_correctly_identified_clusters(df_total, rank, init_routine)
            df["criteria"] = rank
            df["init_routine"] = init_routine
            df_list.append(df)
        df = get_correctly_identified_clusters_by_best_model(df_total, init_routine)
        df["init_routine"] = init_routine
        df["criteria"] = "Best_Model_"
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all

def select_init_routine(df_all, init_routine):
    df = df_all[df_all.init_routine == init_routine]
    df = df.pivot_table(values="correctly_identified_clusters", columns="criteria", index="True cluster number")
    df.columns = df.columns.get_level_values(0)
    df.columns = [''.join(col).strip() for col in df.columns.values]
    df.reset_index(drop=True) # index is now true clusternumber
    df.columns = ["_".join(col.split("_")[:-1]) for col in df.columns]
    return df