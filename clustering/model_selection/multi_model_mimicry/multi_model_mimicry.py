
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
        self.df_scores = df_scores 
        self.model_data = model_data
        # will be set by self.select_candidate_models()
        self.candidate_model_idxs: list[int] = None
        self.M: int = None   # Number of candidate models 
        self.bs_dataset_array : np.ndarray = None
        # will be set by _simulate_parametric_bootstrapped_datasets()
        self.model_clusternumbers = []     
        #will be set by self.run
        self.gofs = None

    def run(self):
        self.gofs = self._cross_fit()
        self.df_gofs = self._process_cross_fit()
        

    def select_candidate_models(self, candidate_model_idx: list[int]) -> list: #TODO
        self.candidate_model_idxs = sorted(candidate_model_idx)
        self.df_scores = self.df_scores.loc[self.df_scores.model_idx.isin(self.candidate_model_idxs)]
        self.M = len(self.candidate_model_idxs)
        self.bs_dataset_array = self._simulate_parametric_bootstrapped_datasets()

    def _simulate_parametric_bootstrapped_datasets(self) -> np.ndarray:
        dataset_array = np.ndarray((self.M, self.params.N_bs, self.M), dtype=np.object_)  # M_model x N_bs x M_data
        for m_data, model_idx in enumerate(self.candidate_model_idxs):
            df_predict = get_prediction_df(self.df_scores, self.model_data, model_idx)
            #df_predict_cleaned = self._clean_outliers(df_predict) solved through soft assignments
            self.model_clusternumbers.append(len(df_predict.prediction_cluster.unique()))
            df_predicted_mixtures = self.calculate_cluster_params_soft(model_idx)
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
            df_scores = emc.df_scores.sort_values(["dataset", self.params.criterion.split("_")[0]+"_rank"])
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

    def calculate_cluster_params_soft(self, model_idx):
        df_predict = get_prediction_df(self.df_scores, self.model_data, model_idx)
        df_n_points = self.get_number_of_points_in_prediction_cluster(df_predict)
        df_cluster_distr_params = self.calculate_cluster_distr_params(self.model_data, model_idx)
        return pd.merge(df_n_points, df_cluster_distr_params, on="prediction_cluster")
    
    @staticmethod
    def calculate_cluster_distr_params(model_data, model_idx):
        inferred_mixture = model_data["inferred_mixtures"][model_idx]
        df_dict = {
            "prediction_cluster": [str(i) for i in range(len(inferred_mixture)//4)],
            "x_mean": [val for val in inferred_mixture[1::4]], 
            "y_mean": [val for val in inferred_mixture[2::4]], 
            "y_std": [val for val in inferred_mixture[3::4]]
            }
        return pd.DataFrame.from_dict(df_dict)

    @staticmethod
    def get_number_of_points_in_prediction_cluster(df_pred: pd.DataFrame) -> pd.DataFrame:
        """derives number of points in one prediction cluster from membership probabilities (=gamma) (soft-assignment)
        * it is assured that the numbers add up to the original point number
        * the numbers can significantly deviate from the hard cluster assignment
        """
        gamma_sum = df_pred[[col for col in df_pred.columns if "gamma" in col]].sum()
        df_n_points_dict = {"prediction_cluster": [], "n_points": []}
        for gamma_pred_cl in gamma_sum.index:
            df_n_points_dict["prediction_cluster"].append(gamma_pred_cl.split("_")[-1])
            df_n_points_dict["n_points"].append(int(gamma_sum[gamma_pred_cl]))
        n_residuals = round(sum(gamma_sum - gamma_sum.astype(int)))
        residuals_sorted = (gamma_sum % gamma_sum.astype(int)).sort_values(ascending=False)
        for i in range(n_residuals):
            gamma_pred_cl = residuals_sorted.index[i].split("_")[-1]
            index_pred_cluster = df_n_points_dict["prediction_cluster"].index(gamma_pred_cl)
            df_n_points_dict["n_points"][index_pred_cluster] += 1
        return pd.DataFrame.from_dict(df_n_points_dict)

    
    def calculate_cluster_params(self, df_predict):
        """outdated; currently mixture of soft and hard assignments used -> make it to hard only to compare perfomance with soft"""
        df_cl_grouped = df_predict.groupby(["prediction_cluster"]).agg({"x": ["mean"], "y": ["mean", "std"]})
        df_cl_grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_cl_grouped.columns.values]
        df_n_points = self.get_number_of_points_in_prediction_cluster(df_predict)
        df_cl_grouped = df_cl_grouped.merge(df_n_points, left_index=True, right_index=True)
        df_cl_grouped = df_cl_grouped.reset_index()
        return df_cl_grouped
    

    def plot_gofs_PCA_components(self, pca_n_components = None):
        if pca_n_components is None:
            pca_n_components = self.params.pca_n_components
        df = self.df_gofs.copy()
        features = [col for col in df.columns if "model_" in col]
        pca = PCA()
        components = pca.fit_transform(df[features])[:, :pca_n_components]
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        df_pca = pd.DataFrame.from_dict({labels[str(i)]: components[:,i] for i in range(pca_n_components) })
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
