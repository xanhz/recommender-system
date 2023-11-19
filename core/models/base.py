from abc import ABC, abstractmethod

import numpy as np
from sklearn import metrics


class LatentFactorModel(ABC):
    def __init__(
        self,
        n_factors: int,
        threshold: float = 0.0001,
        epoch: int = 20,
        verbose_step: int = 5,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.threshold = threshold
        self.epoch = epoch
        self.verbose_step = verbose_step
        self.use_bias = use_bias

        self.observed_set = np.array([])

        self.U = np.array([], dtype=np.float64)
        self.V = np.array([], dtype=np.float64)

        self.n_users = 0
        self.n_items = 0

        self.user_biases = np.array([], dtype=np.float64)
        self.item_biases = np.array([], dtype=np.float64)

        self.global_mean = 0
        self.user_means = np.array([], dtype=np.float64)
        self.item_means = np.array([], dtype=np.float64)

    def _compute_error_matrix(self) -> np.ndarray:
        E = np.zeros(shape=(self.n_users, self.n_items))
        user_ids = self.observed_set[:, 0]
        item_ids = self.observed_set[:, 1]
        ratings = self.observed_set[:, 2]
        E[user_ids, item_ids] = ratings - np.dot(self.U, self.V.T)[user_ids, item_ids]
        return E

    def _compute_rmse(self, E: np.ndarray) -> float:
        mse = np.mean(E ** 2, where=E != 0)
        return np.sqrt(mse)

    def fit(self, rows: np.ndarray) -> 'LatentFactorModel':
        self.observed_set = rows
        self.n_users, self.n_items = rows[:, 0].max() + 1, rows[:, 1].max() + 1

        # Init matrices
        self.U = np.random.randn(self.n_users, self.n_factors)
        self.V = np.random.randn(self.n_items, self.n_factors)

        # Calculate means
        self.global_mean = rows[:, 2].mean()

        self.user_means = np.zeros(shape=(self.n_users, ))
        for user_id in range(self.n_users):
            indices = rows[:, 0] == user_id
            ratings = rows[indices, 2]
            self.user_means[user_id] = 0 if ratings.size == 0 else ratings.mean()

        self.item_means = np.zeros(shape=(self.n_items, ))
        for item_id in range(self.n_items):
            indices = rows[:, 1] == item_id
            ratings = rows[indices, 2]
            self.item_means[item_id] = 0 if ratings.size == 0 else ratings.mean()

        # Calculate biases
        self.user_biases = self.user_means - self.global_mean
        self.item_biases = self.item_biases - self.global_mean

        return self._fit()

    @abstractmethod
    def _fit(self) -> 'LatentFactorModel':
        pass

    def predict_rating(self, user_id: int, item_id: int) -> float:
        predicted_rating = self.U[user_id] @ self.V[item_id]

        if self.use_bias:
            return self.user_biases[user_id] + self.item_biases[item_id] + predicted_rating

        return predicted_rating

    def evaluate(self, test_rows: np.ndarray):
        user_ids = test_rows[:, 0]
        item_ids = test_rows[:, 1]
        expected = test_rows[:, 2]
        map_func = np.vectorize(pyfunc=lambda i, j: self.predict_rating(i, j))
        actual = map_func(user_ids, item_ids)

        print('MAE:', metrics.mean_absolute_error(expected, actual))
        print('MSE:', metrics.mean_squared_error(expected, actual))
        print('RMSE:', metrics.mean_squared_error(expected, actual) ** 0.5)


class UnconstrainedMatrixFactorization(LatentFactorModel):
    def __init__(
        self,
        n_factors: int,
        threshold: float = 0.005,
        epoch: int = 20,
        verbose_step: int = 5,
        regularization: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__(n_factors, threshold, epoch, verbose_step, use_bias)
        self.regularization = regularization
