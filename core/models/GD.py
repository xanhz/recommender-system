import numpy as np

from core.models.base import UnconstrainedMatrixFactorization


class GradientDescent(UnconstrainedMatrixFactorization):
    def __init__(
        self,
        n_factors: int,
        threshold: float = 0.005,
        epoch: int = 20,
        verbose_step: int = 5,
        regularization: float = 0,
        use_bias: bool = False,
        learning_rate: float = 0.005
    ) -> None:
        super().__init__(n_factors, threshold, epoch, verbose_step, regularization, use_bias)
        self.learning_rate = learning_rate

    def _fit(self):
        rmse = np.inf
        epoch = 0

        while epoch < self.epoch:
            E = self._compute_error_matrix()
            rmse = self._compute_rmse(E)
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | RMSE={rmse}')

            if rmse <= self.threshold:
                return self

            gradient_U = -(E @ self.V - self.regularization * self.U)
            gradient_V = -(E.T @ self.U - self.regularization * self.V)

            self.U -= self.learning_rate * gradient_U
            self.V -= self.learning_rate * gradient_V

            if self.use_bias:
                self.U[:, self.n_factors + 1] = 1.0
                self.V[:, self.n_factors] = 1.0

        return self
