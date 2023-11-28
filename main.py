from core import models
from core.dataset import Dataset


def run(dataset_name: str = 'movie-lens-100k', index: int = 1):
    train_data = Dataset.from_csv(
        filepath=f'./datasets/{dataset_name}/{index}/train.csv',
        user_field='UserID',
        item_field='MovieID',
        rating_field='Rating',
    )
    test_data = Dataset.from_csv(
        filepath=f'./datasets/{dataset_name}/{index}/test.csv',
        user_field='UserID',
        item_field='MovieID',
        rating_field='Rating',
    )

    model = models.GradientDescent(
        n_factors=100,
        learning_rate=0.005,
        n_epochs=20,
        regularization=0.06,
        threshold=0.005,
        verbose_step=1,
        use_bias=False,
    )
    model.fit(train_data)
    model.evaluate(test_data)


run(dataset_name='movie-lens-100k', index=1)
