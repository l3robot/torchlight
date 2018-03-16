import abc

import numpy as np

import torch.nn as nn
from torch.autograd import Variable

from .utils import EarlyStopping

    
class BaseTrainer():
    
    def __init__(self, model, optimizer, **kwargs):
        self.model = model
        self.optimizer = optimizer
        ## kwargs assignations
        self.patience = kwargs.get('patience', 5)
        self.cuda = kwargs.get('cuda', False)
        self.verbose = kwargs.get('verbose', False)
        self.simulate_mini_batch = kwargs.get('simulate_mini_batch', False)

    ## deccorator functions
    def isolate_model_mode(validate):
        def new_fct(fct):
            def wrapper(*args, **kwargs):
                model = args[0].model
                last_mode = model.training
                model.eval() if validate is True else model.train()
                out = fct(*args, **kwargs)
                model.train() if last_mode is True else model.eval()
                return out 
            return wrapper
        return new_fct

    ## abstract functions
    @abc.abstractmethod
    def loss_inference(self, inputs, targets):
        return

    ## private functions
    def __update_step(self, inputs, targets):
        self.optimizer.zero_grad()
        loss = self.loss_inference(inputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def __fcuda(self, *tensors):
        return [T.cuda() if self.cuda else T for T in tensors]

    def __one_batch(self, batch, validate):
        inputs, targets = self.__fcuda(*batch)
        if validate:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            loss = self.loss_inference(inputs, targets)
        else:
            inputs = Variable(inputs)
            targets = Variable(targets)
            loss = self.__update_step(inputs, targets)
        return float(loss.data.cpu().numpy()[0])

    def __all_batch(self, dataloader, validate):
        batch_losses = []
        for i_batch, batch in enumerate(dataloader):
            loss = self.__one_batch(batch, validate)
            batch_losses.append(loss)
        # print(np.mean(batch_losses))
        return np.mean(batch_losses)

    @isolate_model_mode(validate=True)
    def __validate_one_epoch(self, validloader):
        valid_loss = self.__all_batch(validloader, validate=True)
        return valid_loss

    @isolate_model_mode(validate=False)
    def __train_one_epoch(self, trainloader):
        train_loss = self.__all_batch(trainloader, validate=False)
        return train_loss

    ## public functions
    def train(self, trainloader, validloader, nb_epochs, early_stopping=False):
        train_losses, valid_losses = [], []
        early_stopper = EarlyStopping(self.patience)
        for i_epoch in range(nb_epochs):
            train_loss = self.__train_one_epoch(trainloader)
            valid_loss = self.__validate_one_epoch(validloader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.verbose:
                print(' [-]: epoch {}, train loss: {}, valid loss: {}'.format(
                    i_epoch+1, train_loss, valid_loss))
            stop = early_stopper.stopping(i_epoch, valid_loss, self.model.state_dict())
            if stop and early_stopping:
                if self.verbose:
                    print(' [-]: early stopping at epoch {}'.format(i_epoch+1))
                break
        self.model.load_state_dict(early_stopper.best_weights)
        return train_losses, valid_losses, early_stopper


class BaseClassificationTrainer(BaseTrainer):

    def __init__(self, model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)

    def loss_inference(self, inputs, targets):
        criterion = nn.CrossEntropyLoss()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss






