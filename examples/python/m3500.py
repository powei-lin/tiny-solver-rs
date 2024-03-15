import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from tiny_solver import GaussNewtonOptimizer, Problem
from tiny_solver.factors import PriorFactor, CostFactorSE2




def load_g2o(file_path: str):
    init_values = {}
    factor_graph = Problem()
    vertex_num = 4000
    with open(file_path) as ifile:
        for line in ifile.readlines():
            items = line[:-1].split(" ")
            if items[0] == "EDGE_SE2":
                if int(items[1]) > vertex_num or int(items[2]) > vertex_num:
                    continue
                point_id0 = f"x{int(items[1])}"
                point_id1 = f"x{int(items[2])}"
                items_float = [float(i) for i in items[3:]]
                dx = items_float[0]
                dy = items_float[1]
                dtheta = items_float[2]
                dpose = np.array([dtheta, dx, dy])

                # if point_id0 == "x10" and point_id1 == "x11":
                #     print(dpose)
                i11, i12, i13, i22, i23, i33 = items_float[3:]
                matrix_i = np.array([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])
                # loss = np.linalg.cholesky(matrix_i)
                factor = CostFactorSE2(dx, dy, dtheta)
                factor_graph.add_residual_block(3, [(point_id0, 3), (point_id1, 3)], factor)
            elif items[0] == "VERTEX_SE2":
                if int(items[1]) > vertex_num:
                    continue
                point_id = f"x{int(items[1])}"
                x = float(items[2])
                y = float(items[3])
                theta = float(items[4])
                # if point_id == "x10" or point_id == "x11":
                #     print(point_id, theta, x, y)

                init_values[point_id] = np.array([theta, x, y], dtype=np.float64)
            else:
                print(items)
                break
    return factor_graph, init_values
    # show_pose(init_values=init_values)
    # print(init_values)


def show_pose(init_values, color):
    data_x = [x[1] for x in init_values.values()]
    data_y = [x[2] for x in init_values.values()]
    plt.scatter(data_x, data_y, s=1, c=color)


def main():
    file_path = "tests/data/input_M3500_g2o.g2o"
    factor_graph, init_values = load_g2o(file_path)

    prior_factor = PriorFactor(np.zeros(3))
    factor_graph.add_residual_block(3, [("x0", 3)], prior_factor)
    solver = GaussNewtonOptimizer()
    # gn = LevenbergMarquardtOptimizer()
    draw = False
    if draw:
        plt.figure(figsize=(8, 8))
        show_pose(init_values, "red")

    start_time = perf_counter()
    solver.optimize(factor_graph, init_values)
    end_time = perf_counter()
    print(f"{solver.__class__.__name__} takes {end_time-start_time:.3f} sec")
    # if draw:
    #     show_pose(init_values, "blue")
    #     ax = plt.gca()
    #     ax.set_xlim((-50, 50))
    #     ax.set_ylim((-80, 20))
    #     plt.tight_layout()
    #     plt.show()
    # print("end")
    pass


if __name__ == "__main__":
    main()
