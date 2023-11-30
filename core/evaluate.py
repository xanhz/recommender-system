from typing import List, Tuple
from core.dataset import Dataset

from core.models.base import LatentFactorModel


class ModelEvaluation:
    def __init__(
        self,
        datasets: List[Tuple[Dataset, Dataset]] = [],
        models: List[LatentFactorModel] = [],
    ) -> None:
        self.datasets = datasets
        self.models = models

    def run(self) -> None:
        for model in self.models:
            print(f'Evaluating model {model.__class__.__name__}')
            for i in range(self.datasets.__len__()):
                print(f'Running test {i + 1}')
                train, test = self.datasets[i]
                model.fit(train)
                model.evaluate(test)

    def add_model(self, model: LatentFactorModel):
        self.models.append(model)
        return self

    def add_dataset(self, dataset: Tuple[Dataset, Dataset]):
        self.datasets.append(dataset)
        return self
