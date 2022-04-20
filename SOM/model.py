from typing import Tuple

import numpy as np
from tqdm import tqdm

from SOM.Distance import distance_functions
from SOM.Neighbor import neighborhood_functions


class SOM:
    def __init__(
            self,
            size: Tuple[int, int],
            feature: int,
            learning_rate: int,
            max_iterations: int,
            shuffle: bool = False,
            neighbor_function: str = "bubble",
            distance_function: str = "euclidean"
    ):
        self.shuffle = shuffle
        self.size = size
        self.feature = feature
        self.learning_rate = learning_rate

        # 处理参数
        self.mutable_update = lambda origin, iteration: origin / (1 + iteration / (max_iterations / 2))
        # 在 [-1, 1] 内生成随机初始权重 x * y * features
        self.weights = np.random.randn(*size, feature) * 2 - 1
        # 初始化激活图
        self.activation_map = np.zeros(size)
        # 初始化网格
        self.x_steps, self.y_steps = np.arange(size[0]), np.arange(size[1])
        self.xx, self.yy = np.meshgrid(self.x_steps, self.y_steps)
        # 初始化距离函数
        self.distance = distance_functions[distance_function]
        self.neighborhood = neighborhood_functions[neighbor_function]

    def fit(self, data: np.ndarray, verbose: bool = True):
        # 训练过程
        batch, feature = data.shape
        assert feature == self.feature, "训练时传入的data维度与设置的不匹配"
        # 开始训练
        for i, x in enumerate(tqdm(data)):
            # 计算全局距离
            self.activation_map = self.distance(x, self.weights)
            # 获取坐标 这里用了一个比较新颖的方法 unravel_index
            winner = np.unravel_index(self.activation_map.argmin(), self.size)
            # 得出更新步长
            eta = self.mutable_update(self.learning_rate, i)
            g = self.neighborhood(winner, 2) * eta
            # 应用更新
            self.weights += np.einsum('ij, ijk->ijk', g, x - self.weights)
