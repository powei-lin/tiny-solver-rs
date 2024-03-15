import numpy as np

class CostFactorSE2:
    def __init__(self, x: float, y: float, theta: float) -> None: ...

class PriorFactor:
    def __init__(self, x: np.ndarray, /) -> None: ...