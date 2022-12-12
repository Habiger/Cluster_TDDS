from dataclasses import dataclass, field

@dataclass
class ExponentialDistribution:
    type: str = "exponential"
    mu: float = None

@dataclass
class NormalDistribution:
    type: str = "normal"
    mu: float = None
    std: float = None

@dataclass
class MixtureComponentDistribution:
    cluster_idx: int
    x: ExponentialDistribution = field(default_factory = lambda: ExponentialDistribution()) # defaults have to be set by default_factory ...
    y: NormalDistribution  = field(default_factory = lambda: NormalDistribution())          # ... else each instance will carry the same object reference
    mix_coef: float = None
    n_points: int = None

    


    