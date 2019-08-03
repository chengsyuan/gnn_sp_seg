import numpy as np
import torch


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor
