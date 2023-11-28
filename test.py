from core import models
from core.dataset import Dataset

train_data = Dataset.from_csv(f'example.csv')
test_data = Dataset.from_csv(f'example.csv')

model = models.StochasticGradientDescent(
    n_factors=3,
    n_epochs=50,
    verbose_step=1,
    use_bias=True,
    threshold=0.005,
    # init_strategy='user_mean',
    # n_workers=6,
    regularization=0.02,
    # learning_rate=0.005
)
model.fit(train_data)
model.evaluate(test_data)
