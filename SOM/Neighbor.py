# Neighborhood Functions
from typing import Tuple

import numpy as np


def bubble(c: Tuple[int, int], sigma: int, x_steps: np.ndarray, y_steps: np.ndarray, **ignore):
    # 更新给定点的上下左右一定范围内的全部点
    ax = np.logical_and(x_steps > c[0] - sigma, x_steps < c[0] + sigma)
    ay = np.logical_and(y_steps > c[1] - sigma, y_steps < c[1] + sigma)
    return np.outer(ax, ay) * 1.


def gaussian(c: Tuple[int, int], sigma: int, xx: np.ndarray, yy: np.ndarray, **ignore):
    # 以C为中心，使用高斯分布计算更新范围
    d = 2 * sigma * sigma
    ax = np.exp(-np.power(xx - xx.T[c], 2) / d)
    ay = np.exp(-np.power(yy - yy.T[c], 2) / d)
    return (ax * ay).T


neighborhood_functions = {
    'bubble': bubble,
    'gaussian': gaussian,
}
