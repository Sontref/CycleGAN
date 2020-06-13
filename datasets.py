import os

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize

# TODO: add transforms
class ImageDataset(Dataset):
    def __init__(self, path_to_data, size=(128, 128), mode='train'):

        self.A = []
        self.B = []

        dir_A = mode + 'A'
        dir_B = mode + 'B'

        for root, _, files in os.walk(path_to_data):
            if root.endswith(dir_A):
                for f in files:
                    img = gray2rgb(imread(os.path.join(root, f)))
                    img = resize(img, size)
                    self.A.append(img)
            elif root.endswith(dir_B):
                for f in files:
                    img = gray2rgb(imread(os.path.join(root, f)))
                    img = resize(img, size)
                    self.B.append(img)

        self.A = np.array(self.A)
        self.B = np.array(self.B)

    def __getitem__(self, index):
        item_A = self.A[index % len(self.A)]
        item_B = self.B[index % len(self.B)]
        return {"A" : item_A, "B" : item_B}

    def __len__(self):
        return max(len(self.A), len(self.B))


class FakeImageBuffer:
    def __init__(self, size=50):
        self.size = size
        self.data = []

    # From https://github.com/eriklindernoren/
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
