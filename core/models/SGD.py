import numpy as np

from core.models.GD import GradientDescent


class StochasticGradientDescent(GradientDescent):
    def _fit(self):
        rmse = np.inf
        epoch = 0

        while epoch < self.n_epochs:
            E = self._compute_error_matrix()
            rmse = self._compute_rmse(E)
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | RMSE={rmse}')

            if rmse <= self.threshold:
                return self

            for i, j, _ in self.dataset.shuffle():
                gradient_Ui = -(E[i, j] * self.V[j] - self.regularization * self.U[i])
                gradient_Vj = -(E[i, j] * self.U[i] - self.regularization * self.V[j])

                self.U[i] -= self.learning_rate * gradient_Ui
                self.V[j] -= self.learning_rate * gradient_Vj

            if self.use_bias:
                self.U[:, -1] = 1.0
                self.V[:, -2] = 1.0

        return self
