import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        lab = self.labels[index]

        return img, lab
