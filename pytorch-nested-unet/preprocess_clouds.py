"""Preprocess data."""
import os
import warnings
from typing import Tuple
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from joblib import Parallel, delayed


def change_color_and_size(image: np.ndarray) -> np.ndarray:
    """画像を読み込んで、白黒に変換."""
    image = rgb2gray(image)
    image = image.astype('float32')
    image = resize(image, (525, 325))
    return image


def convert_pix2img(pix: str) -> np.ndarray:
    """マスクのstrを読み込んで、白黒のマスクに変換."""
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
    if isinstance(mask_rle, str):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        img = img.reshape(shape).T
    else:
        img = np.zeros(shape).T

    return img


def main(folder_name: str, pix_csv_path: str, pix_column: int = 1):
    """Main process in this module.

    folder_name: Anticipating name is 'train_images' or 'test_images' in the folder '../data/'.
    pix_column: マスクのカラム
    """
    paths = glob('../data/' + folder_name + '/*')
    today = datetime.now().strftime('%Y%m%d')
    csv_df = pd.read_csv(pix_csv_path)

    img_folder_path = 'inputs/' + folder_name + '_' + today
    mask_folder_path = 'inputs/' + folder_name + '_mask_' + today

    # Create the folder of save.
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    if not os.path.exists(mask_folder_path):
        os.makedirs(mask_folder_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        def save_process(idx):
            # for image
            path = paths[idx]
            image = imread(path)
            image = change_color_and_size(image)
            file_name = 'img_' + path[-11:-4] + '.npy'
            np.save(img_folder_path + '/' + file_name, image)

            # for mask
            mask_list = []
            for mask_idx in range(4*idx, 4*idx+4):
                mask = convert_pix2img(pix=csv_df.iloc[mask_idx, pix_column])
                mask_list.append(mask)
            mask = np.stack(mask_list)
            file_name = 'msk_' + path[-11:-4] + '.npy'
            np.save(mask_folder_path + '/' + file_name, mask)

        Parallel(n_jobs=-1, verbose=10)([delayed(save_process)(idx) for idx in range(len(paths))])


if __name__ == '__main__':
    main(folder_name='train_images', pix_csv_path='../data/train.csv' )
