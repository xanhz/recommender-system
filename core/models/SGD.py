import numpy as np

from core.models.GD import GradientDescent


class StochasticGradientDescent(GradientDescent):
    def _fit(self):
        rmse = np.inf
        epoch = 0
        global_mean = self.dataset.global_mean if self.use_bias else 0

        while epoch < self.n_epochs and rmse > self.threshold:
            for i, j, r_ij in self.dataset.shuffle():
                r_hat_ij = np.dot(self.U[i, :], self.V[j, :]) + global_mean
                e_ij = r_ij - r_hat_ij

                gradient_Ui = -(e_ij * self.V[j, :] - self.regularization * self.U[i, :])
                gradient_Vj = -(e_ij * self.U[i, :] - self.regularization * self.V[j, :])

                self.U[i, :] -= self.learning_rate * gradient_Ui
                self.V[j, :] -= self.learning_rate * gradient_Vj

            if self.use_bias:
                self.U[:, -1] = 1.0
                self.V[:, -2] = 1.0

            rmse = self._compute_rmse()
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | RMSE={rmse}')

        return self
