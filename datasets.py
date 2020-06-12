import os

import numpy as np
from torch.utils.data import Dataset

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize

class ImageDataset(Dataset):
    def __init__(self, path_to_data, mode='train'):

        self.A = []
        self.B = []

        dir_A = mode + 'A'
        dir_B = mode + 'B'

        for root, _, files in os.walk(path_to_data):
            if root.endswith(dir_A):
                for f in files:
                    self.A.append(gray2rgb(imread(os.path.join(root, f))))
            elif root.endswith(dir_B):
                for f in files:
                    self.B.append(gray2rgb(imread(os.path.join(root, f))))

        self.A = np.array(self.A)
        self.B = np.array(self.B)

    def __getitem__(self, index):
        item_A = self.A[index % len(self.A)]
        item_B = self.B[index % len(self.B)]

        return {"A" : item_A, "B" : item_B}

    def __len__(self):
        return max(len(self.A), len(self.B))
