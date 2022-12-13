from abc import ABC, abstractmethod
from dataclasses import dataclass

from clustering.miscellaneous.parameter_dataclass import Parameter


class Routine(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.params: RoutineParameter

    @abstractmethod
    def algorithm(self):
        pass

    @abstractmethod
    def get_sampled_init_params(self):
        pass

    @abstractmethod
    def get_single_init_param_sample(self):
        pass


@dataclass
class RoutineParameter(Parameter):
    N_cluster_min: int = 1
    N_cluster_max: int = 10
    N_runs_per_clusternumber: int = 10
