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


class CachingDatasetFolder(torchvision.datasets.DatasetFolder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached = {}

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index] 
        else:
            obj = super().__getitem__(index)
            self.cached[index] = obj
            return obj