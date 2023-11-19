from core import models, utils


def run(index: int):
    train_data = utils.read_csv_to_rows(f'./datasets/{index}/train.csv')
    test_data = utils.read_csv_to_rows(f'./datasets/{index}/test.csv')

    model = models.StochasticGradientDescent(n_factors=120, learning_rate=0.00025, epoch=1000)
    model.fit(train_data)
    model.evaluate(test_data)


run(1)
