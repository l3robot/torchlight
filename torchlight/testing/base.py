import abc

import numpy as np
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseTester():

    def __init__(self, model, **kwargs):
        self.model = model
        ## kwargs assignations
        self.cuda = kwargs.get('cuda', False)

    ## deccorator functions
    def isolate_model_mode(validate):
        def new_fct(fct):
            def wrapper(*args, **kwargs):
                model = args[0].model
                last_mode = model.training
                __ = model.eval() if validate else model.train()
                out = fct(*args, **kwargs)
                __ = model.train() if last_mode else model.eval()
                return out
            return wrapper
        return new_fct

    ## private functions
    @abc.abstractmethod
    def loss_acc_inference(self, inputs, targets):
        pass

    def __fcuda(self, tensors):
        return [T.cuda() if self.cuda else T for T in tensors]

    def __one_batch(self, batch):
        inputs, targets = self.__fcuda(batch)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        loss, acc = self.loss_acc_inference(inputs, targets)
        return loss, acc

    ## public functions
    @isolate_model_mode(validate=True)
    def test(self, testloader):
        batch_losses = []
        batch_accs = []
        for batch in testloader:
            loss, acc = self.__one_batch(batch)
            batch_losses.append(loss)
            batch_accs.append(acc)
        return np.mean(batch_losses), np.mean(batch_accs)


class BaseClassificationTester(BaseTester):

    def loss_acc_inference(self, inputs, targets):
        criterion = nn.CrossEntropyLoss()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        numpy_outputs = np.argmax(F.softmax(outputs, dim=1).data.cpu().numpy(), axis=1)
        numpy_targets = targets.data.cpu().numpy()
        numpy_loss = loss.data.cpu().numpy()[0]
        return numpy_loss, accuracy_score(numpy_targets, numpy_outputs)