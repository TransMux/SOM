# Neighborhood Functions
import numpy as np
from scipy import linalg


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return linalg.norm(np.subtract(x, y), axis=-1)


distance_functions = {
    'euclidean': euclidean_distance,
}
