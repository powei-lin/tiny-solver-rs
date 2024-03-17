from typing import List, Tuple, Dict
from enum import Enum

import numpy as np

class Problem:
    def __init__(self) -> None: ...
    def add_residual_block(self, dim_residual: int, variable_key_size_list: List[Tuple[str, int]], factor) -> None: ...

class GaussNewtonOptimizer:
    def __init__(self) -> None: ...
    def optimize(self, problem: Problem, init_values: Dict[str, np.ndarray]) -> None: ...

class LinearSolver(Enum):
    SparseCholesky = ...
    SparseQR = ...

class OptimizerOptions:
    def __init__(
        self,
        max_iteration: int,
        linear_solver_type: LinearSolver,
        verbosity_level: int,
        min_abs_error_decrease_threshold: float,
        min_rel_error_decrease_threshold: float,
        min_error_threshold: float,
    ) -> None: ...
