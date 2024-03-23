import numpy as np
from tiny_solver import Problem, GaussNewtonOptimizer
from tiny_solver.factors import PriorFactor, PyFactor


def cost(x, y, z):
  r0 = x[0] + 2*y[0] + 4*z[0]
  r1 = y[0] * z[0]
  return np.array([r0, r1])

def main():
    problem = Problem()
    pf = PyFactor(cost)
    problem.add_residual_block(2, [('x', 1), ('y', 1), ('z', 1),], pf, None)
    pp = PriorFactor(np.array([3.0]))
    problem.add_residual_block(1, [('x', 1)], pp, None)
    gn = GaussNewtonOptimizer()
    init_values = {"x": np.array([0.7]), "y": np.array([-30.2]), "z": np.array([123.9])}
    result_values = gn.optimize(problem, init_values)
    print(result_values)
    

if __name__ == '__main__':
    main()
