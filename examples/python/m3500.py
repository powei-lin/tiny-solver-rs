import logging
from time import perf_counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tiny_solver import GaussNewtonOptimizer, Problem, OptimizerOptions
from tiny_solver.factors import PriorFactor, BetweenFactorSE2
from tiny_solver.loss_functions import HuberLoss


def load_g2o(file_path: str) -> Tuple[Problem, Dict[str, np.ndarray]]:
    init_values = {}
    factor_graph = Problem()
    with open(file_path) as ifile:
        for line in ifile.readlines():
            items = line[:-1].split(" ")
            if items[0] == "EDGE_SE2":
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
                factor = BetweenFactorSE2(dx, dy, dtheta)
                factor_graph.add_residual_block(3, [(point_id0, 3), (point_id1, 3)], factor, HuberLoss())
            elif items[0] == "VERTEX_SE2":
                point_id = f"x{int(items[1])}"
                x = float(items[2])
                y = float(items[3])
                theta = float(items[4])

                init_values[point_id] = np.array([theta, x, y], dtype=np.float64)
            else:
                print(items)
                break
    return factor_graph, init_values


def show_pose(init_values, color):
    data_x = [x[1] for x in init_values.values()]
    data_y = [x[2] for x in init_values.values()]
    plt.scatter(data_x, data_y, s=1, c=color)


def main():
    FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    file_path = "tests/data/input_M3500_g2o.g2o"
    factor_graph, init_values = load_g2o(file_path)

    prior_factor = PriorFactor(np.zeros(3))
    factor_graph.add_residual_block(3, [("x0", 3)], prior_factor, None)
    solver = GaussNewtonOptimizer()
    # gn = LevenbergMarquardtOptimizer()
    draw = True
    if draw:
        plt.figure(figsize=(8, 8))
        show_pose(init_values, "red")

    optimizer_option = OptimizerOptions(max_iteration=50)
    start_time = perf_counter()
    init_values = solver.optimize(factor_graph, init_values, optimizer_option)
    end_time = perf_counter()
    print(f"{solver.__class__.__name__} takes {end_time-start_time:.3f} sec")
    if draw:
        show_pose(init_values, "blue")
        ax = plt.gca()
        ax.set_xlim((-50, 50))
        ax.set_ylim((-80, 20))
        plt.tight_layout()
        plt.savefig("m3500_py.png")
    print("end")


if __name__ == "__main__":
    main()
