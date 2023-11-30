from abc import ABC, abstractmethod

import numpy as np

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

    def _compute_error_matrix(self):
        user_ids = self.dataset.user_ids
        item_ids = self.dataset.item_ids
        global_mean = self.dataset.global_mean if self.use_bias else 0

        R = self.dataset.rating_matrix
        R_hat = self.U @ self.V.T + global_mean

        E = np.zeros(self.dataset.shape)
        E[user_ids, item_ids] = R[user_ids, item_ids] - R_hat[user_ids, item_ids]

        return E

    def _compute_rmse(self) -> float:
        E = self._compute_error_matrix()
        mse = np.mean(E ** 2, where=E != 0)
        return np.sqrt(mse)

    def _init_matrices(self, **kwargs):
        n_users, n_items = self.dataset.shape
        n_factors = self.n_factors

        rng = np.random.RandomState()
        U = rng.normal(0, 0.1, (n_users, n_factors))
        V = rng.normal(0, 0.1, (n_items, n_factors))

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
        n_users, n_items = self.dataset.shape

        if user_id >= n_users or item_id >= n_items:
            return self.dataset.global_mean

        predicted = np.dot(self.U[user_id, :], self.V[item_id, :])

        if self.use_bias:
            predicted += self.dataset.global_mean

        return predicted if not clip else np.clip(predicted, *self.dataset.rating_range)

    def evaluate(self, test_set: Dataset):
        user_ids = test_set.user_ids
        item_ids = test_set.item_ids
        map_func = np.vectorize(pyfunc=lambda i, j: self.predict_rating(i, j))
        predicted = map_func(user_ids, item_ids)
        return test_set.evaluate(predicted)

    def make_recommendation_for_user(self, user_id: int, n_items: int = 10):
        ratings = self.U[user_id, :] @ self.V.T

        item_ids = np.argsort(ratings)[::-1]  # Sort indices in descending order
        sorted_ratings = np.array(list(zip(item_ids, ratings[item_ids])))

        rated_item_ids = self.dataset.rated_items_by_user(user_id)

        unrated = sorted_ratings[~np.isin(sorted_ratings[:, 0], rated_item_ids)]
        k = max(len(unrated), n_items)

        return unrated[:k]

    def make_recommendation_for_item(self, item_id: int, n_users: int = 10):
        ratings = self.U @ self.V[item_id, :].T

        user_ids = np.argsort(ratings)[::-1]  # Sort indices in descending order
        sorted_ratings = np.array(list(zip(user_ids, ratings[user_ids])))

        rated_user_ids = self.dataset.users_rate_item(item_id)

        unrated = sorted_ratings[~np.isin(sorted_ratings[:, 0], rated_user_ids)]
        k = max(len(unrated), n_users)

        return unrated[:k]


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
