import time
from functools import partial
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm

from SOM.Distance import distance_functions
from SOM.Neighbor import neighborhood_functions


class SOM:
    def __init__(
            self,
            size: Tuple[int, int],
            feature: int,
            learning_rate: int or float,
            max_iterations: int,
            shuffle: bool = False,
            neighbor_function: str = "bubble",
            distance_function: str = "euclidean"
    ):
        self.shuffle = shuffle
        self.size = size
        self.feature = feature
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.errors = []

        # 处理参数
        self.mutable_update = lambda origin, iteration: origin
        # 在 [-1, 1] 内生成随机初始权重 x * y * features
        self.weights = np.random.randn(*size, feature) * 2 - 1
        # 初始化激活图
        self.activation_map = np.zeros(size)
        # 初始化网格
        self.x_steps, self.y_steps = np.arange(size[0]), np.arange(size[1])
        self.xx, self.yy = np.meshgrid(self.x_steps, self.y_steps)
        # 初始化距离函数
        self.distance = distance_functions[distance_function]

        self.neighborhood = partial(
            neighborhood_functions[neighbor_function],
            x_steps=self.x_steps,
            y_steps=self.y_steps,
            xx=self.xx,
            yy=self.yy
        )

    def fit(self, data: np.ndarray, verbose: bool = True):
        # 训练过程
        batch, feature = data.shape
        assert feature == self.feature, "训练时传入的data维度与设置的不匹配"
        # 开始训练
        for _ in range(self.max_iterations):
            if self.shuffle:
                np.random.shuffle(data)
            for i, x in enumerate(tqdm(data, desc=f"Epoch {_}")):
                winner = self.get_winner(x)
                # 得出更新步长
                eta = self.mutable_update(self.learning_rate, i)
                g = self.neighborhood(winner, 4) * eta
                # 应用更新
                self.weights += np.einsum('ij, ijk->ijk', g, x - self.weights)
            time.sleep(0.01)
            print(f"Epoch {_} Error: ", self.map_error(data))
            time.sleep(0.01)

    def get_winner(self, x):
        # 计算全局距离
        self.activation_map = self.distance(x, self.weights)
        # 获取坐标 这里用了一个比较新颖的方法 unravel_index
        winner = np.unravel_index(self.activation_map.argmin(), self.size)
        return winner

    def map_error(self, data: np.ndarray):
        # data      batch * features
        # weights   x * y * features
        # Distance  batch * (x * y)
        # distance = np.einsum("ij, xyj -> ixy", data, self.weights).reshape(data.shape[0], -1) Wrong !
        # Or : distance = np.linalg.norm(data[:, None, ...] - self.weights.reshape(-1,64)[None, ...], axis=-1)
        distance = self._distance_from_weights(data)
        coords = np.argmin(distance, axis=1)
        weights = self.weights[np.unravel_index(coords, self.size)]
        # Error     batch * 1
        error = np.linalg.norm(data - weights, axis=1).mean()
        self.errors.append(error)
        return error

    def _distance_from_weights(self, data):
        # 实现两个不同长度矩阵欧氏距离的计算
        # https://stackoverflow.com/questions/60686539/python-euclidean-distance-different-size-vectors
        weights_flat = self.weights.reshape(-1, self.weights.shape[2])
        input_data_sq = np.power(data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(data, weights_flat.T)
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
