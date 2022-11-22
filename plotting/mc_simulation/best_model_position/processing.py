import pandas as pd


def get_position_of_best_models(df_total):
    ranks = [col for col in df_total.columns if "rank" in col]
    df_list, df_list_rank_cluster, df_list_position_inside_clusternumber = [], [], []
    for init_routine in df_total.init_routine.unique():
        df_ir = df_total[df_total.init_routine == init_routine]
        result_dict_position, result_dict_rank_cluster, result_dict_position_inside_clusternumber = {}, {}, {}
        for rank in ranks:
            positions, ranks_cluster, positions_inside_clusternumber = [], [], []
            for dataset in df_ir.dataset.unique():
                df = df_ir[df_ir.dataset == dataset]
                df = df.sort_values(rank)
                # position
                position = df.number_identified_cluster.argmax()
                positions.append(position)
                # rank_of_cluster
                rank_of_cluster = len(df.iloc[[i for i in range(position)], :].N_cluster.unique())
                ranks_cluster.append(rank_of_cluster)
                # position inside specific clusternumber
                clusternumber = df.iloc[position, df.columns.get_loc("N_cluster")]
                position_inside_clusternumber = df[df.N_cluster == clusternumber].number_identified_cluster.argmax()
                positions_inside_clusternumber.append(position_inside_clusternumber)
            result_dict_position[rank], result_dict_rank_cluster[rank], result_dict_position_inside_clusternumber[rank] = positions, ranks_cluster, positions_inside_clusternumber
        result_dict_position["init_routine"], result_dict_rank_cluster["init_routine"], result_dict_position_inside_clusternumber["init_routine"] = [init_routine] * len(df_ir.dataset.unique()), [init_routine] * len(df_ir.dataset.unique()), [init_routine] * len(df_ir.dataset.unique())
        df_list.append(pd.DataFrame.from_dict(result_dict_position))
        df_list_rank_cluster.append(pd.DataFrame.from_dict(result_dict_rank_cluster))
        df_list_position_inside_clusternumber.append(pd.DataFrame.from_dict(result_dict_position_inside_clusternumber))
    return pd.concat(df_list), pd.concat(df_list_rank_cluster), pd.concat(df_list_position_inside_clusternumber)
