from typing import List, Tuple, Dict, Optional
from enum import Enum

import numpy as np

from tiny_solver.factors import Factor
from tiny_solver.loss_functions import Loss

class Problem:
    def __init__(self) -> None: ...
    def add_residual_block(
        self, dim_residual: int, variable_key_size_list: List[Tuple[str, int]], factor: Factor, loss: Optional[Loss]
    ) -> None: ...

class GaussNewtonOptimizer:
    def __init__(self) -> None: ...
    def optimize(
        self, problem: Problem, init_values: Dict[str, np.ndarray], optimizer_options: Optional[OptimizerOptions] = None
    ) -> None: ...

class LinearSolver(Enum):
    SparseCholesky = ...
    SparseQR = ...

class OptimizerOptions:
    def __init__(
        self,
        max_iteration: int = 100,
        linear_solver_type: LinearSolver = LinearSolver.SparseCholesky,
        verbosity_level: int = 0,
        min_abs_error_decrease_threshold: float = 1e-5,
        min_rel_error_decrease_threshold: float = 1e-5,
        min_error_threshold: float = 1e-8,
    ) -> None: ...

def first_derivative_test(f: callable, x): ...