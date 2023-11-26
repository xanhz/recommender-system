import numpy as np
from sklearn.decomposition import TruncatedSVD

from core import initstrategies
from core.models.base import LatentFactorModel


class SingularValueDecomposition(LatentFactorModel):
    def __init__(
        self,
        n_factors: int,
        n_epochs: int = 20,
        threshold: float = 0.005,
        verbose_step: int = 5,
        use_bias: bool = False,
        init_strategy: initstrategies.StrategyType = 'random',
    ) -> None:
        super().__init__(n_factors, n_epochs, threshold, verbose_step, use_bias)
        self.init_strategy = init_strategy

    def _compute_objective_function_value(self, E: np.ndarray) -> float:
        return np.sum(np.abs(E))

    def _fit(self):
        R = self.dataset.to_matrix()
        Rf = initstrategies.create(self.init_strategy).fill(R)
        n_factors = self.n_factors
        unobserved_set = R == 0

        diff = np.inf
        epoch = 0
        while diff > self.threshold and epoch < self.n_epochs:
            svd = TruncatedSVD(n_components=n_factors)
            self.U = svd.fit_transform(Rf)
            self.V = svd.components_.transpose()

            R_approx = self.U @ self.V.transpose()
            D = np.where(unobserved_set, R_approx - Rf, 0)
            diff = np.sum(np.abs(D))
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | Diff={diff}')

            Rf[unobserved_set] = R_approx[unobserved_set]

        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return self.U[user_id] @ self.V[item_id].T
