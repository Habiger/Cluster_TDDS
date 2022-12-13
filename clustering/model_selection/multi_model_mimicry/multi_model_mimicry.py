
"""work in progress
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA

from postanalysis.post_processing_true_models import get_prediction_df

from clustering.miscellaneous.parameter_dataclass import Parameter, nested_dataclass
from clustering.input_data.simulated_experiment_class import SimulatedExperiment
from clustering.model_selection.criteria.scoreboard import create_scoreboard
from clustering.model_selection.criteria.functions import criteria_dict
from clustering.clustering import EMClusteringParameter, EMClustering





@nested_dataclass
class MultiModelMimicryParameter(Parameter):
    N_bs: int = 3 # number of bootstraps
    criterion: str = "ll_score"     # Goodness of Fit Measure (GOF)
    n_neighbors: int = 10
    pca_n_components: int = 2
    N_runs_per_clusternumber: int = 3  # for EMClusteringParameter
    emc: EMClusteringParameter = EMClusteringParameter(**{
        "parallel": {
            "n_jobs": 10
        },
        "em": {
            "max_reiterations": 5000,
            "em_algorithm": {
                "em_tol": 1e-3
            }
        }
    })  # needed M times because every model has its own clusternumber


class MultiModelMimicry:
    def __init__(self, df_scores, model_data, **kwargs):
        self.params = MultiModelMimicryParameter(**kwargs)
        self.candidate_model_idxs = self._select_candidate_models(df_scores, model_data)
        self.df_scores = df_scores.loc[df_scores.model_idx.isin(self.candidate_model_idxs)] # only candidate models will be analysed
        self.model_data = model_data        
        self.M = len(self.candidate_model_idxs)   # Number of candidate models 
        self.model_clusternumbers = [] # will be set by _simulate_parametric_bootstrapped_datasets()    
        self.bs_dataset_array = self._simulate_parametric_bootstrapped_datasets()
        #will be set by self.run
        self.gofs = None

    def run(self):
        self.gofs = self._cross_fit()
        self.df_gofs = self._process_cross_fit()
        

    def _select_candidate_models(self, df_scores, model_data) -> list: #TODO
        return sorted([13725, 13710, 13757])

    def _simulate_parametric_bootstrapped_datasets(self) -> np.ndarray:
        dataset_array = np.ndarray((self.M, self.params.N_bs, self.M), dtype=np.object_)  # M_model x N_bs x M_data
        for m_data, model_idx in enumerate(self.candidate_model_idxs):
            df_predict = get_prediction_df(self.df_scores, self.model_data, model_idx)
            df_predict_cleaned = self._clean_outliers(df_predict)
            self.model_clusternumbers.append(len(df_predict_cleaned.prediction_cluster.unique()))
            df_predicted_mixtures = self.calculate_cluster_params(df_predict_cleaned)
            dataset_array[:,:,m_data] = np.array([SimulatedExperiment.from_parametric_bootstrap(df_predicted_mixtures, f"{_i}999{model_idx}") for _i in range(self.params.N_bs)])[:]
        return dataset_array

    @staticmethod
    def _clean_outliers(df_pred):
        df_pred_count_points = df_pred.groupby(["prediction_cluster"]).agg({"x": "count"}).reset_index() 
        useful_cluster = df_pred_count_points[df_pred_count_points.x >1].prediction_cluster
        df_pred_cleaned = df_pred[df_pred.prediction_cluster.isin(useful_cluster)]
        return df_pred_cleaned

    def _get_predicted_clusternumbers(self): # currently in _simulate_parametric_bootstrapped_datasets
        return [self.df_scores.loc[self.df_scores.model_idx == model_idx, "N_cluster"].values[0] for model_idx in self.candidate_model_idxs] 
            

    def _cross_fit(self):
        gofs = np.ndarray(shape=(self.M, self.params.N_bs, self.M))
        for m in range(self.M):
            emc_params = EMClusteringParameter(**self.params.emc.get_dict())
            emc_params.cluster_init.routine.N_cluster_min = int(self.model_clusternumbers[m])
            emc_params.cluster_init.routine.N_cluster_max = int(self.model_clusternumbers[m])
            emc = EMClustering(**emc_params.get_dict())
            emc.load_experiments(self.bs_dataset_array[m, :, :].reshape(-1))
            emc.run()
            df_results, model_data = emc.df_results, emc.model_data
            df_scores, df_scores_nan = create_scoreboard(df_results, model_data)
            df_scores = df_scores.sort_values(["dataset", self.params.criterion.split("_")[0]+"_rank"])
            gof_results = df_scores.groupby(["dataset"])[self.params.criterion].first()
            gofs[m, :, :] = np.array(gof_results).reshape((self.params.N_bs ,self.M))[:, :]
        return gofs

    def _process_cross_fit(self):
        df_list = []
        for m in range(self.M):
            df = pd.DataFrame(self.gofs[:, :, m].T, columns = [f"model_{m_}" for m_ in self.candidate_model_idxs])
            df["data_model"] = f"data_model_{self.candidate_model_idxs[m]}"
            df_list.append(df)
        df_obs = self.get_observed_gof()
        df_obs["data_model"] = "obs"
        df = pd.concat(df_list + [df_obs])
        df = df.reset_index(drop=True)
        return df

    def get_observed_gof(self):
        df = self.df_scores
        df_obs = pd.DataFrame.from_dict({f"model_{m}": list(df.loc[df.model_idx == m, self.params.criterion]) for m in self.candidate_model_idxs})
        return df_obs


    def plot_gofs(self):
        alpha_models, alpha_obs = 0.25, 1
        df_plot = self.df_gofs
        g = sns.PairGrid(df_plot, hue="data_model", hue_kws={"alpha": [alpha_models for i in self.candidate_model_idxs]+[alpha_obs]})
        g = g.map_diag(sns.kdeplot, shade=True)
        g = g.map_lower(plt.scatter)
        g = g.map_upper(plt.scatter)
        g = g.add_legend()
        fig = plt.gcf()
        fig.set_size_inches(25,15)
        # vertical grey line on diagonal for observed gof for better visual inspecting
        axs = [ax for ax in fig.axes if ax.get_ylabel() == "Density"]
        for i, ax in enumerate(axs):
            ax.axvline(x=df_plot[df_plot.data_model == "obs"].iloc[0, i], color="grey", alpha=0.8)
        plt.close()
        return fig

    def predict_model(self):
        df = self.df_gofs.copy()
        X = df.loc[df.data_model != "obs", ~df.columns.isin(["data_model"])].to_numpy()
        y = df.loc[df.data_model != "obs", df.columns.isin(["data_model"])].to_numpy()
        X_obs = df.loc[df.data_model == "obs", ~df.columns.isin(["data_model"])].to_numpy()
        knn = KNN(n_neighbors=self.params.n_neighbors)
        knn.fit(X, y)
        self.knn_model_probabilities = knn.predict_proba(X_obs)
        return knn.predict(X_obs)
    
    @staticmethod
    def calculate_cluster_params(df):
        df_cl_grouped = df.groupby(["prediction_cluster"]).agg({"x": ["count","mean"], "y": ["mean", "std"]})
        df_cl_grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_cl_grouped.columns.values]
        df_cl_grouped = df_cl_grouped.reset_index().rename(columns={"x_count": "n_points"})  
        return df_cl_grouped

    def plot_gofs_PCA_components(self):
        df = self.df_gofs.copy()
        features = [col for col in df.columns if "model_" in col]
        pca = PCA()
        components = pca.fit_transform(df[features])[:, :self.params.pca_n_components]
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        df_pca = pd.DataFrame.from_dict({labels[str(i)]: components[:,i] for i in range(self.params.pca_n_components) })
        df_pca["data_model"] = df["data_model"]
        alpha_models, alpha_obs = 0.25, 1
        df_plot = df_pca
        g = sns.PairGrid(df_plot, hue="data_model", hue_kws={"alpha": [alpha_models]*(len(df.columns)-1) +[alpha_obs]})
        g = g.map_diag(sns.kdeplot, shade=True)
        g = g.map_lower(plt.scatter)
        g = g.map_upper(plt.scatter)
        g = g.add_legend()
        fig = plt.gcf()
        fig.set_size_inches(25,15)
        # vertical grey line on diagonal for observed gof for better visual inspecting
        axs = [ax for ax in fig.axes if ax.get_ylabel() == "Density"]
        for i, ax in enumerate(axs):
            ax.axvline(x=df_plot[df_plot.data_model == "obs"].iloc[0, i], color="grey", alpha=0.8)
        plt.close()
        return fig
