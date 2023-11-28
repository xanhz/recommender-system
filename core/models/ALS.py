from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from core.models.base import UnconstrainedMatrixFactorization


class AlternatingLeastSquares(UnconstrainedMatrixFactorization):
    def __init__(
        self,
        n_factors: int,
        n_epochs: int = 20,
        threshold: float = 0.005,
        verbose_step: int = 5,
        use_bias: bool = False,
        regularization: float = 0,
        n_workers: int = None,
    ) -> None:
        super().__init__(n_factors, n_epochs, threshold, verbose_step, use_bias, regularization)
        self.n_workers = n_workers

    def _fit(self):
        R = self.dataset.rating_matrix
        n_users, n_items = R.shape
        n_factors = self.n_factors

        I = np.eye(n_factors)
        rmse = np.inf
        epoch = 0

        def solve_Ui(i: int):
            item_ids = self.dataset.rated_items_by_user(i)

            sub_R = R[i, item_ids]
            sub_V = self.V[item_ids, :]

            A_i = sub_V.T @ sub_V + self.regularization * I
            B_i = sub_V.T @ sub_R.T

            return np.linalg.solve(A_i, B_i)

        def solve_Vj(j: int):
            user_ids = self.dataset.users_rate_item(j)

            sub_R = R[user_ids, j]
            sub_U = self.U[user_ids, :]

            A_j = sub_U.T @ sub_U + self.regularization * I
            B_j = sub_U.T @ sub_R

            return np.linalg.solve(A_j, B_j)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            while epoch < self.n_epochs:
                rmse = self._compute_rmse()
                epoch += 1

                if epoch % self.verbose_step == 0:
                    print(f'Epoch={epoch} | RMSE={rmse}')

                if rmse <= self.threshold:
                    return self

                user_futures = [executor.submit(solve_Ui, i) for i in range(n_users)]
                wait(user_futures)
                self.U = np.array([future.result() for future in user_futures])

                item_futures = [executor.submit(solve_Vj, j) for j in range(n_items)]
                wait(item_futures)
                self.V = np.array([future.result() for future in item_futures])

        return self
