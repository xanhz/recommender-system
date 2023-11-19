import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def read_csv_to_rows(
    filepath: str,
    user_field: str = 'user_id',
    item_field: str = 'item_id',
    rating_field: str = 'rating'
) -> np.ndarray:
    df = pd.read_csv(filepath)

    df = df[[user_field, item_field, rating_field]]
    rows = df.to_numpy()
    rows[:, 0] -= 1
    rows[:, 1] -= 1

    return rows


def from_rows_to_matrix(rows: np.ndarray, missing_fill: float = None) -> np.ndarray:
    user_ids = rows[:, 0]
    item_ids = rows[:, 1]
    ratings = rows[:, 2]

    sparse_matrix = coo_matrix(
        (ratings, (user_ids, item_ids)),
        shape=(np.max(user_ids) + 1, np.max(item_ids) + 1),
        dtype=np.float64,
    )

    rating_matrix = sparse_matrix.toarray()

    if missing_fill is not None:
        rating_matrix[rating_matrix == 0] = missing_fill

    return rating_matrix
