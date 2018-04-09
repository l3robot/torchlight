import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .base import BaseLoop


class BaseTesterLoop(BaseLoop):

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        ## kwargs assignations
        self.cuda = kwargs.get('cuda', False)

    ## public functions
    @BaseLoop.isolate_model_mode(validate=True)
    def test(self, testloader):
        loss = self._BaseLoop__all_batch(testloader, validate=True)
        metrics = self.compute_metrics()
        return (loss, metrics)
