from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from core.models.base import UnconstrainedMatrixFactorization
from core.utils import from_rows_to_matrix


class AlternatingLeastSquares(UnconstrainedMatrixFactorization):
    def __init__(
        self,
        n_factors: int,
        threshold: float = 0.005,
        epoch: int = 20,
        verbose_step: int = 5,
        regularization: float = 0,
        use_bias: bool = False,
        n_workers: int = None
    ) -> None:
        super().__init__(n_factors, threshold, epoch, verbose_step, regularization, use_bias)
        self.n_workers = n_workers

    def _fit(self):
        R = from_rows_to_matrix(self.observed_set)
        I = np.eye(self.n_factors)
        rmse = np.inf
        epoch = 0

        def solve_Ui(i: int):
            rated_items_by_user = self.observed_set[self.observed_set[:, 0] == i]
            item_ids = rated_items_by_user[:, 1]

            sub_R = R[i, item_ids]
            sub_V = self.V[item_ids, :]

            A_i = sub_V.T @ sub_V + self.regularization * I
            B_i = sub_V.T @ sub_R.T

            return np.linalg.solve(A_i, B_i)

        def solve_Vj(j: int):
            rated_users_by_item = self.observed_set[self.observed_set[:, 1] == j]
            user_ids = rated_users_by_item[:, 0]

            sub_R = R[user_ids, j]
            sub_U = self.U[user_ids, :]

            A_j = sub_U.T @ sub_U + self.regularization * I
            B_j = sub_U.T @ sub_R

            return np.linalg.solve(A_j, B_j)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            while epoch < self.epoch:
                E = self._compute_error_matrix()
                rmse = self._compute_rmse(E)
                epoch += 1

                if epoch % self.verbose_step == 0:
                    print(f'Epoch={epoch} | RMSE={rmse}')

                if rmse <= self.threshold:
                    return self

                user_futures = [executor.submit(solve_Ui, i) for i in range(self.n_users)]
                wait(user_futures)
                self.U = np.array([future.result() for future in user_futures])

                item_futures = [executor.submit(solve_Vj, j) for j in range(self.n_items)]
                wait(item_futures)
                self.V = np.array([future.result() for future in item_futures])

        return self
