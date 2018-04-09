import numpy as np
import torchvision


class CachingImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached = {}

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index] 
        else:
            img = super().__getitem__(index)
            self.cached[index] = img
            return img

    def load_all(self):
        data = [self[i] for i in range(len(self))]
        return data

    def compute_labels_reweighting(self):
        train_data, train_targets = list(zip(*self.load_all()))
        _, idx, stats = np.unique(train_targets, return_inverse=True, return_counts=True)
        stats = stats / len(train_targets)
        stats = 1/(len(stats)*stats)
        return stats[idx] / np.sum(stats[idx])


# class CachingDatasetFolder(torchvision.datasets.DatasetFolder):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.cached = {}

#     def __getitem__(self, index):
#         if index in self.cached:
#             return self.cached[index] 
#         else:
#             obj = super().__getitem__(index)
#             self.cached[index] = obj
#             return obj