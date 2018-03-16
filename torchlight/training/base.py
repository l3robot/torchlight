import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    
class BaseTrainer():
    
    def __init__(self, model, optimizer, **kwargs):
        self.model = model
        self.optimizer = optimizer
        ## kwargs assignations
        self.cuda = kwars.get('cuda', False)
        self.verbose = kwars.get('verbose', False)
        self.simulate_mini_batch = kwars.get('simulate_mini_batch', False)

    @staticmethod
    def isolate_model_mode(fct, validate):
        def new_fct(*args, **kwargs):
            last_model_mode = self.model.training
            self.model.eval() if validate is True else model.train()
            fct(*args, **kwargs) 

    @abc.abstractmethod
    def inference(self, inputs, targets):
        return

    def __update_step(self, inputs, targets):
        self.optimizer.zero_grad()
        loss = self.one_batch_inference(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def __fcuda(self, tensors):
        return [T.cuda() for T in tensors if self.cuda else T]

    def __one_batch(self, batch, validate):
        inputs, targets = self.__fcuda(*batch)
        if validate:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            loss = self.inference(inputs, targets)
        else:
            inputs = Variable(inputs)
            targets = Variable(targets)
            loss = self.__update_step(inputs, targets)
        return float(loss.data.cpu().numpy()[0])

    def __one_epoch(self, dataloader, validate):
        batch_losses = []
        for i_batch, batch in enumerate(dataloader):
            loss = self.__one_batch(batch, validate)
            batch_losses.append(loss)
        return np.mean(batch_losses)

    def __validate(self, validloader):
        last_model_mode = self.model.training
        self.model.eval()
        valid_loss = self._one_epoch(validloader, validate=True)
        self.model.training = last_model_mode
        return valid_loss
