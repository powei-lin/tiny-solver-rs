import tiny_solver
import numpy as np

print(tiny_solver.sum_as_string(1, 2))
tiny_solver.mult(np.zeros((1, 2)))
print(tiny_solver.__version__)
print(dir(tiny_solver))
a = tiny_solver.Dual64()
print(a.first_derivative)
b = tiny_solver.FactorSE2(1.0, 2.0, 3.0)
print(dir(b))
print(b.dtheta)
c = tiny_solver.Problem()
print(dir(c))
c.num = 200
print(c.num)
c.add_residual_block()