import numpy as np
from tiny_solver import Problem, GaussNewtonOptimizer
from tiny_solver.factors import PriorFactor, PyFactor

# define custom cost function in python
def cost(x: np.ndarray, yz: np.ndarray) -> np.ndarray:
    r0 = x[0] + 2 * yz[0] + 4 * yz[1]
    r1 = yz[0] * yz[0]
    return np.array([r0, r1])


def main():

    # initialize problem (factor graph)
    problem = Problem()

    # factor defined in python
    custom_factor = PyFactor(cost)
    problem.add_residual_block(
        2,
        [
            ("x", 1),
            ("yz", 2),
        ],
        custom_factor,
        None,
    )

    # prior factor import from rust
    prior_factor = PriorFactor(np.array([3.0]))
    problem.add_residual_block(1, [("x", 1)], prior_factor, None)

    # initial values
    init_values = {"x": np.array([0.7]), "yz": np.array([-30.2, 123.4])}

    # optimizer
    optimizer = GaussNewtonOptimizer()
    result_values = optimizer.optimize(problem, init_values)

    # result
    for k, v in result_values.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
