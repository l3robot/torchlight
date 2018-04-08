import torchvision


class CachingImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, *args, **kwargs):
        super(CachingImageFolder, self).__init__(*args, **kwargs)
        self.cached = {}

    def __getitem__(self, index):
        if index in self.cached:
            return self.cached[index] 
        else:
            img = super(CachingImageFolder, self).__getitem__(index)
            self.cached[index] = img
            return img