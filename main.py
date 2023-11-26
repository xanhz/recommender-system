from core import models
from core.dataset import Dataset

def run(index: int):
    train_data = Dataset.from_csv(f'./datasets/{index}/train.csv')
    test_data = Dataset.from_csv(f'./datasets/{index}/test.csv')

    model = models.StochasticGradientDescent(n_factors=120, learning_rate=0.00025, epoch=1000)
    model.fit(train_data)
    model.evaluate(test_data)


run(1)
