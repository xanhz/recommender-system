import numpy as np
from core import models, utils, initstrategies

train_data = utils.read_csv_to_rows(f'example.csv')
test_data = utils.read_csv_to_rows(f'example.csv')

model = models.AlternatingLeastSquares(
    n_factors=2,
    epoch=4,
    verbose_step=5,
    regularization=0.01
)
model.fit(train_data)
model.evaluate(test_data)
