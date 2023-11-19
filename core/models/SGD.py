import numpy as np

from core.models.GD import GradientDescent


class StochasticGradientDescent(GradientDescent):
    def _fit(self):
        S = self.observed_set[:, [0, 1]]
        rmse = np.inf
        epoch = 0

        while rmse > self.threshold and epoch < self.epoch:
            np.random.shuffle(S)
            E = self._compute_error_matrix()
            rmse = self._compute_rmse(E)
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | RMSE={rmse}')

            if rmse <= self.threshold:
                return self

            for i, j in S:
                gradient_Ui = -(E[i, j] * self.V[j] -
                                self.regularization * self.U[i])
                gradient_Vj = -(E[i, j] * self.U[i] -
                                self.regularization * self.V[j])

                self.U[i] -= self.learning_rate * gradient_Ui
                self.V[j] -= self.learning_rate * gradient_Vj

                if self.use_bias:
                    self.U[i, self.n_factors + 1] = 1.0
                    self.V[j, self.n_factors] = 1.0

        return self
