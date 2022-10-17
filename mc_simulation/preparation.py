from copy import deepcopy


def initialize_results_dict(init_routines = ['OPTICS', 'random_inside']):
    """initializes results dict for parameter analysis

    Args:
        init_routines (list, optional): which cluster initilization routines to use. Defaults to ['OPTICS', 'random_inside'].

    Returns:
        dict: empty dictionary with predefinied entries for the different types of data to be stored in the analysis run
    """
    sub_results = {
        "init_params": [],
        "df_scores": [],
        "df_scores_na": [],
    }
    run_results = {init: deepcopy(sub_results) for init in init_routines}
    return run_results