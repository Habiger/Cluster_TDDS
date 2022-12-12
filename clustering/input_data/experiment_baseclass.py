import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Experiment(ABC):
    """Represents the data of one experiment. Instances of this class will be used to pass experimental/simulated data to the algorithms. \\
        * contains at least an array containing the data (self.X) and its corresponding ID (self.id)
    """

    @abstractmethod
    def __init__(self):
        self.id: str           # unique identifier for the datasets

        #TODO implement X as a getter vs implement df as a getter?! or better: choose one datatype!
        self.df: pd.DataFrame  # dataframe containing at least columns "x", "y"
        self.X: np.ndarray     # array with two columns: 
                               #    first column (x): exponential distributed coordinates
                               #    second column (y): normal distributed coordinates


        
