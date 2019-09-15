"""Dataset."""
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """マスクの生成→データセット"""
    def __init__(self,
                 img_paths: List[np.array],
                 mask_paths: List[np.array],
                 transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

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
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
       
        return image, mask


class RandomCrop(object):
    """Random Crop."""
    def __init__(self, crop_size=(256, 256)):
        self.crop_size = crop_size

    def __call__(self, image):
        _, width, height = image.shape

        # 0~(400-224)の間で画像のtop, leftを決める
        left = np.random.randint(0, width - self.crop_size[1])
        top = np.random.randint(0, height - self.crop_size[0])

        # top, leftから画像のサイズである224を足して、bottomとrightを決める
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]

        # 決めたtop, bottom, left, rightを使って画像を抜き出す
        image = image[:, left:right, top:bottom]
        return image


class Resize(object):
    def __init__(self, resize_size=(256, 256)):
        self.resize_size = resize_size

    def __call__(self, image):
        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, self.resize_size,
                           interpolation=cv2.INTER_AREA)
        if len(image.shape) == 2:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = image.transpose(2, 0, 1)
        return image
