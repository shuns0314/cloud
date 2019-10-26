"""Preprocess data."""
import os
import warnings
import argparse
from typing import Tuple
from glob import glob
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma
from joblib import Parallel, delayed


def parse_args():
    """Parser."""
    today = datetime.now().strftime('%Y%m%d')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default=today,
                        help='name of folder to be created.\
                              save format: \
                              inputs/image_ARGS.NAME and mask_ARGS.NAME\
                              defalut: TODAY',
                        type=str)
    parser.add_argument('--images_folder_path',
                        help='folder path of images to be preprocessed',
                        type=str)
    parser.add_argument('--mask',
                        help='path of csv for encoded mask')
    parser.add_argument('--mask_column',
                        default=1,
                        help='mask column of dataframe')
    parser.add_argument('--glay',
                        default=False,
                        help='change color to gray')
    parser.add_argument('--resize',
                        default=(325, 525),
                        type=Tuple[int, int],
                        help='resize')
    parser.add_argument('--gamma',
                        default=1,
                        type=float,
                        help='adjust_gamma')
    args = parser.parse_args()
    return args


def change_color_and_size(args, image: np.ndarray) -> np.ndarray:
    """画像を読み込んで、白黒に変換."""
    if args.glay:
        image = rgb2gray(image)

    image = image.astype('float32')
    image = resize(image, args.resize)
    return image


def convert_pix2img(args, mask: str) -> np.ndarray:
    """マスクのstrを読み込んで、マスク画像に変換."""
    mask = rle_decode(mask, (1400, 2100))
    mask = mask.astype('float32')
    mask = resize(mask, args.resize)

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

        for low, high in zip(starts, ends):
            img[low:high] = 1
        img = img.reshape(shape, order='F')
    else:
        img = np.zeros(shape, order='F')

    return img


def create_onehotlabel(mask_df: pd.DataFrame) -> pd.DataFrame:
    """Count the clouds in the mask to create a data frame."""
    split_df = mask_df['Image_Label'].str.split('_', n=1, expand=True)
    mask_df["Image"] = split_df[0]
    mask_df["Label"] = split_df[1]
    mask_df["encode"] = mask_df.apply(lambda x: 1 if isinstance(x["EncodedPixels"]) is str else 0, axis=1)
    climate_df = mask_df[["Image", "Label", "encode"]].groupby(["Image", 'Label']).sum().unstack(level=1)
    climate_df.columns = climate_df.columns.droplevel(level=0)
    return climate_df


def main():
    """Main process in this module.

    folder_name: Anticipating name is 'train_images' or 'test_images' in the folder '../data/'.
    pix_column: マスクのカラム
    """
    args = parse_args()
    paths = glob(args.images_folder_path + '/*')

    mask_column = args.mask_column

    img_folder_path = f'inputs/image_{args.name}'
    mask_folder_path = f'inputs/mask_{args.name}'

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
            image = change_color_and_size(args, image)
            image = adjust_gamma(image, gamma=args.gamma)
            file_name = 'img_' + path[-11:-4] + '.npy'
            np.save(img_folder_path + '/' + file_name, image)

            # for mask
            if args.mask != "None":
                csv_df = pd.read_csv(args.mask)
                mask_list = []
                for mask_idx in range(4*idx, 4*idx+4):
                    mask = convert_pix2img(args, mask=csv_df.iloc[mask_idx, mask_column])
                    # クロージング処理（マスクの穴を消す）
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    mask_list.append(mask)
                mask = np.stack(mask_list)
                file_name = 'msk_' + path[-11:-4] + '.npy'
                np.save(mask_folder_path + '/' + file_name, mask)

        Parallel(n_jobs=-1, verbose=10)([delayed(save_process)(idx) for idx in range(len(paths))])


if __name__ == '__main__':
    main()
