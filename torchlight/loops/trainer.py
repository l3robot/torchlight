import numpy as np

from .base import BaseLoop
from .utils import EarlyStopping


class BaseTrainerLoop(BaseLoop):

    def __init__(self, model, optimizer, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        ## kwargs assignations
        self.patience = kwargs.get('patience', 5)
        self.cuda = kwargs.get('cuda', False)
        self.verbose = kwargs.get('verbose', False)
        self.simulate_mini_batch = kwargs.get('simulate_mini_batch', False)

    ## public functions
    def train(self, trainloader, validloader, nb_epochs, early_stopping=False):
        self.show_metrics()
        train_losses, valid_losses = [], []
        early_stopper = EarlyStopping(self.patience)
        for i_epoch in range(nb_epochs):
            train_loss = self._BaseLoop__train_one_epoch(trainloader)
            valid_loss = self._BaseLoop__validate_one_epoch(validloader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.verbose:
                print(' [-]: epoch {}, train loss: {:.4f}, valid loss: {:.4f}'\
                    .format(i_epoch+1, train_loss, valid_loss, self.show_metrics()))
            stop = early_stopper.stopping(i_epoch, valid_loss, self.model.state_dict())
            if stop and early_stopping:
                if self.verbose:
                    print(' [-]: early stopping at epoch {}'.format(i_epoch+1))
                break
        self.model.load_state_dict(early_stopper.best_weights)
        return train_losses, valid_losses, early_stopper
