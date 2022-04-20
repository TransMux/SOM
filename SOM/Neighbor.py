# Neighborhood Functions
from typing import Tuple

import numpy as np


def bubble(c: Tuple[int, int], sigma: int, x_steps: np.ndarray, y_steps: np.ndarray):
    # 更新给定点的上下左右一定范围内的全部点
    ax = np.logical_and(x_steps > c[0] - sigma, x_steps < c[0] + sigma)
    ay = np.logical_and(y_steps > c[1] - sigma, y_steps < c[1] + sigma)
    return np.outer(ax, ay) * 1.


neighborhood_functions = {
    'bubble': bubble,
}
