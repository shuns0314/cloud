"""Dataset."""
from typing import Tuple, List
import argparse

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy.ndimage.interpolation import rotate

from data_augmentation import *

class Dataset(torch.utils.data.Dataset):
    """マスクの生成→データセット"""
    def __init__(self,
                 args,
                 img_paths: List[np.array],
                 mask_paths: List[np.array],
                 train=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.train = train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        resize = Resize()

        image = np.load(img_path)
        image = image.astype('float32') / 255
        if len(image.shape) == 2:
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
        image = image.transpose(2, 0, 1) # (channel, width, height) に変換
        image = resize(image)

        mask = np.load(mask_path) # (channel, width, height) になってる
        mask = mask.astype('float32')
        mask = resize(mask)

        # 普通にdatasetのtransformでimageとmaskをランダムでtransformかけようとすると、
        # imageとmaskそれぞれにrandomがかかるっぽい。
        if self.args.aug and self.train:
            if np.random.random() < 0.5:
                holizon = HorizontalFlip()
                image = holizon(image)
                mask = holizon(mask)

            if np.random.random() < 0.5:
                vertical = VerticalFlip()
                image = vertical(image)
                mask = vertical(mask)

            if np.random.random() < 0.5:
                angle = np.random.randint(*(0, 180))
                image = image.transpose(1, 2, 0) # channelを後ろに持ってくる
                image = rotate(image, angle)
                mask = mask.transpose(1, 2, 0)
                mask = rotate(mask, angle)
 
                image = resize(image.transpose(2, 0, 1))
                mask = resize(mask.transpose(2, 0, 1))
            
            # Random crop
            if np.random.random() < 0.5:
                _, width, height = image.shape
                crop_size = (192, 192)
                left = np.random.randint(0, width - crop_size[1])
                top = np.random.randint(0, height - crop_size[0])

                bottom = top + crop_size[0]
                right = left + crop_size[1]

                image = image[:, left:right, top:bottom]
                image = resize(image)
                mask = mask[:, left:right, top:bottom]
                mask = resize(mask)

            if np.random.random() < 0.5:
                _, width, height = 

        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())
        return image, mask
