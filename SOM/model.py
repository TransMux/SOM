from typing import Tuple

import numpy as np


class SOM:
    def __init__(self, size: Tuple[int, int], feature: int, learning_rate, max_iterations, shuffle=False):
        self.shuffle = shuffle
        self.size = size
        self.feature = feature
        self.learning_rate = learning_rate

        # 处理参数
        self.mutable_update = lambda origin, iteration: origin / (1 + iteration / (max_iterations / 2))
        # 在 [-1, 1] 内生成随机初始权重
        self.weights = np.random.randn(*size) * 2 - 1
        # 初始化激活图
        self.activation_map = np.zeros(size)
        # 初始化网格
        self.x_steps, self.y_steps = np.arange(size[0]), np.arange(size[1])
        self.xx, self.yy = np.meshgrid(self.x_steps, self.y_steps)

    def fit(self):
        pass
