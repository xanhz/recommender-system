import numpy as np


def random_uniform_matrix(row: int, col: int, low: float = 0, high: float = 1):
    rng = np.random.default_rng()
    return rng.uniform(low, high, (row, col))
