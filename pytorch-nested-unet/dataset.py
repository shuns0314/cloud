"""Dataset."""
from typing import Tuple, List

import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """マスクの生成→データセット"""
    def __init__(self,
                 img_paths: List[np.array],
                 mask_paths: List[np.array]):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.load(img_path)
        image = image.astype('float32')
        if len(image.shape) == 2:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])

        mask = np.load(mask_path)
        mask = mask.astype('float32')

        return image, mask
