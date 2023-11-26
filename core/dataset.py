from typing import Any, Generator, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn import metrics


class Dataset:
    def __init__(
        self,
        rows: np.ndarray,
        rating_range: Tuple[float, float] = None,
        shape: Tuple[int, int] = None
    ) -> None:
        self.rows = rows
        self.n_records = len(rows)

        self.shape: Tuple[int, int] = shape
        if shape is None:
            self.shape = (rows[:, 0].max() + 1, rows[:, 1].max() + 1)

        ratings = rows[:, 2]
        self.rating_range: Tuple[float, float] = rating_range
        if rating_range is None:
            self.rating_range = (ratings.min(), ratings.max())

        self.global_mean: float = ratings.mean()

    @staticmethod
    def from_csv(
        filepath: str,
        user_field: str = 'user_id',
        item_field: str = 'item_id',
        rating_field: str = 'rating'
    ) -> 'Dataset':
        df = pd.read_csv(filepath)

        df = df[[user_field, item_field, rating_field]]
        rows = df.to_numpy()
        rows[:, 0] -= 1
        rows[:, 1] -= 1

        return Dataset(rows)

    @property
    def user_ids(self) -> np.ndarray:
        """Extract user column in dataset"""
        return self.rows[:, 0].astype(np.uint32)

    @property
    def item_ids(self) -> np.ndarray:
        """Extract item column in dataset"""
        return self.rows[:, 1].astype(np.uint32)

    @property
    def ratings(self) -> np.ndarray:
        """Extract rating column in dataset"""
        return self.rows[:, 2]

    def all(self) -> Generator[Tuple[int, int, float], Any, Any]:
        for i, j, r_ij in self.rows[:, [0, 1, 2]]:
            yield i, j, r_ij

    def shuffle(self) -> Generator[Tuple[int, int, float], Any, Any]:
        np.random.shuffle(self.rows)
        return self.all()

    def evaluate(self, predicted: np.ndarray) -> None:
        expected = self.rows[:, 2]
        self.rows = np.column_stack((self.rows[:, [0, 1, 2]], predicted))
        print('MAE:', metrics.mean_absolute_error(expected, predicted))
        print('MSE:', metrics.mean_squared_error(expected, predicted))
        print('RMSE:', metrics.mean_squared_error(expected, predicted) ** 0.5)

    def to_matrix(self) -> np.ndarray:
        user_ids = self.user_ids
        item_ids = self.item_ids
        ratings = self.ratings
        shape = self.shape

        sparse_matrix = coo_matrix(
            (ratings, (user_ids, item_ids)),
            shape=shape,
            dtype=np.float64,
        )

        return sparse_matrix.toarray()

    def rated_items_by_user(self, user_id: int):
        records = self.rows[self.user_ids == user_id]
        return np.unique(records[:, 1])

    def users_rate_item(self, item_id: int):
        records = self.rows[self.item_ids == item_id]
        return np.unique(records[:, 0])
