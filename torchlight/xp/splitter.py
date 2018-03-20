import copy
import math
from random import Random

import numpy as np


def idx_splitter(size, prop, random_state=42):
    scope_random = Random(random_state)
    idx = list(range(size))
    scope_random.shuffle(idx)
    split = math.floor(prop*size)
    fold1_idx, fold2_idx = idx[:split], idx[split:]
    return fold1_idx, fold2_idx


def train_test_splitter(dataset, train_prop=0.8, random_state=42):
    ## chossing the idx
    train_idx, test_idx = idx_splitter(len(dataset), train_prop, random_state)    
    ## train dataset
    train_dataset = copy.deepcopy(dataset)
    train_dataset.train_data = train_dataset.train_data[train_idx]
    train_dataset.train_labels = np.array(train_dataset.train_labels)[train_idx]
    ## test dataset
    test_dataset = copy.deepcopy(dataset)
    test_dataset.train_data = test_dataset.train_data[test_idx]
    test_dataset.train_labels = np.array(test_dataset.train_labels)[test_idx]
    return train_dataset, test_dataset
