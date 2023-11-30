import numpy as np
from sklearn.decomposition import TruncatedSVD

from core.models.base import LatentFactorModel


class SingularValueDecomposition(LatentFactorModel):
    def _fit(self):
        R = self.dataset.rating_matrix
        Rf = self._mean_centering(R)

        mask = R == 0

        diff = np.inf
        epoch = 0
        while epoch < self.n_epochs and diff > self.threshold:
            svd = TruncatedSVD(n_components=self.n_factors)
            self.U = svd.fit_transform(Rf)
            self.V = svd.components_.transpose()

            R_approx = self.U @ svd.components_
            D = np.where(mask, R_approx - Rf, 0)
            diff = np.sum(np.abs(D))
            epoch += 1

            if epoch % self.verbose_step == 0:
                print(f'Epoch={epoch} | Diff={diff}')

            Rf[mask] = R_approx[mask]

        return self

    def _mean_centering(self, R: np.ndarray):
        def mean_centering_row(row: np.ndarray):
            available_indices = row != 0

            if not np.any(available_indices):
                return np.full(row.shape, 0.0)

            row[~available_indices] = row.mean(where=available_indices)

            return row

        return np.apply_along_axis(mean_centering_row, axis=1, arr=R)
