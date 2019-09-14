"""Preprocess data."""
import os
import warnings
from typing import Tuple
from glob import glob
from datetime import datetime

import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import cv2
import pandas as pd


def change_color_and_size(image: np.ndarray) -> np.ndarray:
    # 画像を読み込んで、白黒に変換
    image = rgb2gray(image)
    image = image.astype('float32') / 255
    image = resize(image, (525, 325))
    return image


def convert_pix2img(pix: str) -> np.ndarray:
    # マスクのstrを読み込んで、白黒のマスクにヘナ間
    mask = rle_decode(pix, (2100, 1400))
    mask = mask.astype('float32')
    mask = resize(mask, (525, 325)) 

    return mask


def rle_decode(mask_rle: str, shape: Tuple[int, int]):
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


def main(folder_name: str, pix_csv_path: str, pix_column:int = 1):
    """Main process in this module.

    folder_name: Anticipating name is 'train_images' or 'test_images' in the folder '../data/'.
    pix_column: マスクのカラム
    """
    paths = glob('../data/'+folder_name+'/*')
    today = datetime.now().strftime('%Y%m%d')
    csv_df = pd.read_csv(pix_csv_path)

    
    # Create save folder
    if not os.path.exists('inputs/' + folder_name + today):
        os.makedirs('inputs/' + folder_name + today)
    if not os.path.exists('inputs/' + folder_name + today):#mask?
        os.makedirs('inputs/' + folder_name + today)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for idx in tqdm(range(len(paths))):
            path = paths[idx]
            image = imread(path)
            image = change_color_and_size(image)
            np.save('img_'path[-11:-4]+'.npy', image)

            mask_list = []
            for mask_idx in range(4*idx, idx+4):
                mask = convert_pix2img(csv_df.iloc[pix_column, mask_idx])
                mask_list.append(mask)
            mask = np.stack(mask_list)
            np.save('msk_'path[-11:-4]+'.npy'mask)
                


if __name__ == '__main__':
    main()
