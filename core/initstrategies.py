from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

StrategyType = Literal['random', 'item_mean', 'user_mean', 'mean']


class Strategy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fill(self, R: np.ndarray) -> np.ndarray:
        pass


class RandomStrategy(Strategy):
    def fill(self, R: np.ndarray) -> np.ndarray:
        observed_set = R != 0
        unobserved_set = ~observed_set
        n_missing_values = np.count_nonzero(unobserved_set)
        min_ = R.min(where=observed_set, initial=np.inf)
        max_ = R.max(where=observed_set, initial=-np.inf)
        Rf = R.copy()
        Rf[unobserved_set] = np.random.rand(n_missing_values) * (max_ - min_) + min_
        return Rf


class UserMeanStrategy(Strategy):
    def fill(self, R: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.__replace_by_row_mean, axis=1, arr=R)

    def __replace_by_row_mean(self, row: np.ndarray):
        copied = row.copy()
        available_indices = copied != 0

        if not np.any(available_indices):
            return np.full(row.shape, 0.0)

        copied[~available_indices] = row.mean(where=available_indices)
        return copied


class ItemMeanStrategy(Strategy):
    def fill(self, R: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.__replace_by_col_mean, axis=0, arr=R)

    def __replace_by_col_mean(self, col: np.ndarray):
        copied = col.copy()
        available_indices = copied != 0

        if not np.any(available_indices):
            return np.full(col.shape, 0.0)

        copied[~available_indices] = col.mean(where=available_indices)
        return copied


class MeanStrategy(Strategy):
    def fill(self, R: np.ndarray) -> np.ndarray:
        global_mean = R.mean(where=R != 0)
        Rf = R.copy()
        Rf[Rf == 0] = global_mean
        return Rf


def create(strategy: StrategyType) -> Strategy:
    if strategy == 'random':
        return RandomStrategy()
    if strategy == 'user_mean':
        return UserMeanStrategy()
    if strategy == 'item_mean':
        return ItemMeanStrategy()
    if strategy == 'mean':
        return MeanStrategy()
    raise Exception('Invalid strategy')
