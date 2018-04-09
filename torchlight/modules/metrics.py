import abc

import numpy as np
from sklearn.metrics import accuracy_score


class BaseMetric():

    def __init__(self):
        self.preds = []
        self.targets = []

    def append(self, preds, targets):
        self.preds.extend(preds.data.cpu().numpy())
        self.targets.extend(targets.data.cpu().numpy())

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError

    def show(self):
        return '{}: {:.4f}'.format(self.name, self.compute())


class AccuracyScore(BaseMetric):

    def __init__(self):
        super().__init__()
        self.name = 'accuracy'

    def compute(self):
        preds = np.argmax(self.preds, axis=1)
        return accuracy_score(preds, self.targets)