from abc import ABC, abstractmethod

import numpy as np

from core import utils
from core.dataset import Dataset


class LatentFactorModel(ABC):
    def __init__(
        self,
        n_factors: int,
        n_epochs: int = 20,
        threshold: float = 0.005,
        verbose_step: int = 5,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.verbose_step = verbose_step
        self.use_bias = use_bias
        self.dataset: Dataset = None
        self.U: np.ndarray = None
        self.V: np.ndarray = None

    @property
    def R_hat(self):
        return self.U @ self.V.T + self.dataset.global_mean

    def _compute_error_matrix(self):
        user_ids = self.dataset.user_ids
        item_ids = self.dataset.item_ids
        ratings = self.dataset.ratings
        shape = self.dataset.shape

        E = np.zeros(shape)

        E[user_ids, item_ids] = ratings - self.R_hat[user_ids, item_ids]

        return E

    def _init_matrices(self, **kwargs):
        n_users, n_items = self.dataset.shape
        n_factors = self.n_factors

        U = utils.random_uniform_matrix(n_users, n_factors)
        V = utils.random_uniform_matrix(n_items, n_factors)

        if self.use_bias:
            U = np.column_stack([
                U,
                np.full(shape=(n_users, ), fill_value=0.0),
                np.full(shape=(n_users, ), fill_value=1.0),
            ])

            V = np.column_stack([
                V,
                np.full(shape=(n_items, ), fill_value=1.0),
                np.full(shape=(n_items, ), fill_value=0.0),
            ])

        return U, V

    def _compute_rmse(self, E: np.ndarray) -> float:
        mse = np.mean(E ** 2, where=E != 0)
        return np.sqrt(mse)

    def fit(self, dataset: Dataset) -> 'LatentFactorModel':
        self.dataset = dataset
        U, V = self._init_matrices()
        self.U = U
        self.V = V
        return self._fit()

    @abstractmethod
    def _fit(self) -> 'LatentFactorModel':
        pass

    def predict_rating(self, user_id: int, item_id: int, clip: bool = True) -> float:
        predicted = self.dataset.global_mean + self.U[user_id] @ self.V[item_id].T
        return predicted if not clip else np.clip(predicted, *self.dataset.rating_range)

    def evaluate(self, test_set: Dataset):
        user_ids = test_set.user_ids
        item_ids = test_set.item_ids
        map_func = np.vectorize(pyfunc=lambda i, j: self.predict_rating(i, j))
        predicted = map_func(user_ids, item_ids)
        return test_set.evaluate(predicted)


class UnconstrainedMatrixFactorization(LatentFactorModel):
    def __init__(
        self,
        n_factors: int,
        n_epochs: int = 20,
        threshold: float = 0.005,
        verbose_step: int = 5,
        use_bias: bool = False,
        regularization: float = 0.0
    ) -> None:
        super().__init__(n_factors, n_epochs, threshold, verbose_step, use_bias)
        self.regularization = regularization
