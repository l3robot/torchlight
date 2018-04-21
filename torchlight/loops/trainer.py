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
        self.saving_path = kwargs.get('saving_path', '.')
        self.cuda = kwargs.get('cuda', False)
        self.verbose = kwargs.get('verbose', False)
        self.simulate_mini_batch = kwargs.get('simulate_mini_batch', False)

    def get_state(self):
        state = {'weights': copy.deepcopy(self.model.state_dict()),
                 'train_losses': self.train_losses,
                 'valid_losses': self.valid_losses,
                 'early_stopper': self.early_stopper}
        return state

    def load_state(self, state):
        self.model.load_state_dict(state['weights'])
        self.train_losses = state['train_losses']
        self.valid_losses = state['valid_losses']
        self.early_stopper = state['early_stopper']

    def save_sate(self):
        with open(os.path.join(self.saving_path, 'checkpoint.pkl') , 'wb') as f:
            pkl.dump(self.get_state, f)

    ## public functions
    def train(self, trainloader, validloader, nb_epochs, early_stopping=False, checkpoint=None):
         if checkpoint is not None:
            self.load_state(checkpoint)
        else:
            self.train_losses, self.valid_losses = [], []
            self.early_stopper = EarlyStopping(self.patience)
        for i_epoch in range(nb_epochs):
            train_loss = self._BaseLoop__train_one_epoch(trainloader)
            valid_loss = self._BaseLoop__validate_one_epoch(validloader)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            if self.verbose:
                print(' [-]: epoch {}, train loss: {:.4f}, valid loss: {:.4f}, {}'\
                    .format(i_epoch+1, train_loss, valid_loss, self.show_metrics()))
            stop = self.early_stopper.stopping(i_epoch, valid_loss, self.model.state_dict())
            self.save_state()
            if stop and early_stopping:
                if self.verbose:
                    print(' [-]: early stopping at epoch {}'.format(i_epoch+1))
                break
        self.model.load_state_dict(self.early_stopper.best_weights)
        return train_losses, valid_losses, early_stopper
