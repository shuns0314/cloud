"""Dataset."""
from typing import Tuple

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """マスクの生成→データセット"""
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple(np.ndarray, np.ndarray):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像を読み込んで、白黒に変換
        image = rgb2gray(imread(img_path))
        image = image.astype('float32') / 255
        image = resize(image, (525, 325))

        # マスクのstrを読み込んで、白黒のマスクにヘナ間
        mask = self.rle_decode(mask_path[idx], (2100, 1400))
        mask = mask.astype('float32')
        mask = resize(mask, (525, 325)) 

        return image, mask

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        return img.reshape(shape).T
