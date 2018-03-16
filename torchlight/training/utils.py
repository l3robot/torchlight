import copy

import numpy as np


class EarlyStopping(object):
    
    def __init__(self, patience=5):
        self.patience = patience
        self.miss = 0
        self.best_epoch = 0
        self.best_valid = np.inf
        self.best_weights = None

    def stopping(self, epoch, valid_loss, weights):
        if self.best_valid > valid_loss:
            self.miss = 0
            self.best_epoch = epoch
            self.best_valid = valid_loss
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.miss == self.patience - 1:
            self.miss += 1
            return True
        else:
            self.miss += 1
            return False