import tiny_solver
from tiny_solver.factors import BetweenFactor, CostFactorSE2
import numpy as np

print(tiny_solver.__version__)
print(dir(tiny_solver))
print(tiny_solver.sum_as_string(1, 2))
tiny_solver.mult(np.zeros((1, 2)))
a = tiny_solver.Dual64()
print(a.first_derivative)
b = CostFactorSE2(1.0, 2.0, 3.0)
# print("factor module\n", dir(factors))
# b = Costf(1.0, 2.0, 3.0)

print(type(b))
print(dir(b))
print(b)
c = tiny_solver.Problem()
print(dir(c))
c.num = 200
print(c.num)
d = BetweenFactor(np.array([2.0,3.0]))
c.add_residual_block(1, [("aa", 1)], d)
c.add_residual_block(1, [("aaa", 1)], b)
# c.add_residual_block(1, [("aaa", 1)])
# c.add_residual_block(1, [("aa", 1)])
# d = tiny_solver.BetweenFactor()
# d.ttt()
# tiny_solver.te(d)