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
        image = image.astype('float32') / image.max()
        if len(image.shape) == 2:
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
        image = image.transpose(2, 0, 1) # (channel, width, height) に変換
        image = resize(image)

        mask = np.load(mask_path) # (channel, width, height) になってる
        mask = mask.astype('uint8')
        mask = resize(mask)

        # 普通にdatasetのtransformでimageとmaskをランダムでtransformかけようとすると、
        # imageとmaskそれぞれにrandomがかかるっぽい。
        if self.args.aug and self.train:
            # Data augmentation of position
            # Don't augment data of twenty percent.
            random_num = np.random.random()

            # Random crop or Move
            if random_num < 0.5:
                move = Move(rate=0.6, max_move_rate=0.1)
                image, mask = move(image, mask)
            else:
                random_crop = RandomCrop(rate=0.6, crop_size=(192, 192))
                image, mask = random_crop(image, mask)

            random_num = np.random.random()

            if random_num > 0.8:
                # HorizonFlip
                horizon_flip = RandomFlip(axis="horizon", rate=0.6)
                image, mask = horizon_flip(image, mask)


                # VerticalFlip
                vertical_flip = RandomFlip(axis="vertical", rate=0.6)
                image, mask = vertical_flip(image, mask)

            elif random_num > 0.6:
                # Rotation
                random_rotate = RandomRotation(rate=0.8, angle_range=(0, 30))
                image, mask = random_rotate(image, mask)

            elif random_num > 0.4:
                # Bounding only horizon or vertical flip
                bounding_only = BoundingOnlyDA(rate=0.8, classes=self.args.n_classes)
                image, mask = bounding_only(image, mask)

            elif random_num > 0.2:
                # CutOff
                cutoff = CutOff(mask_size=(50, 100), rate=0.8)
                image, mask = cutoff(image, mask)

            # Data augmentation of brightness
            # adjust_gamma = Adjust_gamma(rate=0.7, gamma_range=(0.7, 1))
            # image = adjust_gamma(image)

            # Equalize_hist
            # equalize = Equalize(rate=0.2)
            # image = equalize(image)


        image = torch.from_numpy(image.copy().astype(np.float32))
        mask = torch.from_numpy(mask.copy().astype(np.float32))

        return image, mask
