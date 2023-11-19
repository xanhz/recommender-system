import numpy as np
from sklearn.decomposition import TruncatedSVD

from core import initstrategies, utils
from core.models.base import LatentFactorModel


class SingularValueDecomposition(LatentFactorModel):
    def __init__(
        self,
        n_factors: int,
        threshold: float = 0.005,
        epoch: int = 20,
        verbose_step: int = 5,
        use_bias: bool = False,
        init_strategy: initstrategies.StrategyType = 'random'
    ) -> None:
        super().__init__(n_factors, threshold, epoch, verbose_step, use_bias)
        self.init_strategy = init_strategy

    def _compute_objective_function_value(self, E: np.ndarray) -> float:
        return np.sum(np.abs(E))

    def _fit(self):
        R = utils.from_rows_to_matrix(self.observed_set)
        Rf = initstrategies.create(self.init_strategy).fill(R)
        unobserved_set = R == 0

        diff = np.inf
        epoch = 0
        while diff > self.threshold and epoch < self.epoch:
            svd = TruncatedSVD(n_components=self.n_factors)
            self.U = svd.fit_transform(Rf)
            self.V = svd.components_.transpose()

            R_approximated = self.U @ self.V.transpose()
            D = np.where(unobserved_set, R_approximated - Rf, 0)
            diff = np.sum(np.abs(D))
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | Diff={diff}')

            Rf[unobserved_set] = R_approximated[unobserved_set]

        return self
