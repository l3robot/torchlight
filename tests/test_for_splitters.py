import os
import math
import unittest

import torch
import torchvision

from .context import torchlight
from torchlight.xp import idx_splitter, mnist_train_test_splitter


class TestBasicSplitters(unittest.TestCase):
    
    def test_right_size(self):
        mnist_path = '.mnist'
        cifar10_path = '.cifar10'
        ## mnist test init
        proportion = 0.8
        self.mnist = torchvision.datasets.MNIST(download=True, train=True, root=mnist_path)
        self.mnist_train, self.mnist_test = mnist_train_test_splitter(self.mnist, proportion)
        self.mnist_train_size = math.floor(proportion * len(self.mnist))
        self.mnist_test_size = len(self.mnist) - self.mnist_train_size
        # cifar test init
        proportion = 0.9
        self.cifar10 = torchvision.datasets.CIFAR10(download=True, train=True, root=cifar10_path)
        self.cifar10_train, self.cifar10_test = mnist_train_test_splitter(self.cifar10, proportion)
        self.cifar10_train_size = math.floor(proportion * len(self.cifar10))
        self.cifar10_test_size = len(self.cifar10) - self.cifar10_train_size
        ## test
        self.assertEqual(len(self.mnist_train), self.mnist_train_size)
        self.assertEqual(len(self.mnist_test), self.mnist_test_size)
        self.assertEqual(len(self.cifar10_train), self.cifar10_train_size)
        self.assertEqual(len(self.cifar10_test), self.cifar10_test_size)
        ## delete
        os.system('rm -r {}'.format(mnist_path))
        os.system('rm -r {}'.format(cifar10_path))

    def test_idx_size(self):
        size = 100
        proportion = 0.8
        fold1_idx, fold2_idx = idx_splitter(size, prop=proportion, random_state=42)
        self.assertEqual(len(fold1_idx), math.floor(proportion*size))
        self.assertEqual(len(fold2_idx), size-math.floor(proportion*size))
        proportion = 0.9
        fold1_idx, fold2_idx = idx_splitter(size, prop=proportion, random_state=42)
        self.assertEqual(len(fold1_idx), math.floor(proportion*size))
        self.assertEqual(len(fold2_idx), size-math.floor(proportion*size))

    def test_idx_redundancy(self):
        fold1_idx, fold2_idx = idx_splitter(100, prop=0.8, random_state=42)
        self.assertEqual(len(set(fold1_idx)), len(fold1_idx))
        self.assertEqual(len(set(fold2_idx)), len(fold2_idx))

    def test_idx_overlap(self):
        fold1_idx, fold2_idx = idx_splitter(100, prop=0.8, random_state=42)
        self.assertEqual(len(set(fold1_idx) & set(fold2_idx)), 0)

    def test_idx_stochasticity(self):
        fold1_idx1, fold2_idx1 = idx_splitter(100, prop=0.8, random_state=42)
        fold1_idx2, fold2_idx2 = idx_splitter(100, prop=0.8, random_state=43)
        self.assertNotEqual(set(fold1_idx1), set(fold1_idx2))
        self.assertNotEqual(set(fold2_idx1), set(fold2_idx2))        


if __name__ == '__main__':
    unittest.main()
