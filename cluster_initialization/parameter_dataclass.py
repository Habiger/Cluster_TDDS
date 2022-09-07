from dataclasses import dataclass

@dataclass(eq=False)
class Distribution:
    type: str
    mu: float = None
    std: float = None

@dataclass
class Params:
    cluster: int
    mix_coef: float = None
    n_points: int = None
    x: Distribution = None
    y: Distribution = None
    
    def __post_init__(self):
        # needs to be done here else every instance of Params has an identical instance of the Distiribution class
        self.x = Distribution("exponential")
        self.y = Distribution("normal")



    