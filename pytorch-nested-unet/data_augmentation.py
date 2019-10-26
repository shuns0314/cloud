"""Data Augmentation."""
from typing import Tuple

import numpy as np
from scipy.ndimage.interpolation import rotate
import skimage
import cv2


class RandomCrop(object):
    """Random Crop."""
    def __init__(self,
                 crop_size: Tuple[int] = (192, 192),
                 rate: int = 0.5) -> None:
        self.crop_size = crop_size
        self.rate = rate

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.rate:
            _, width, height = image.shape
            left = np.random.randint(0, width - self.crop_size[1])
            top = np.random.randint(0, height - self.crop_size[0])

            bottom = top + self.crop_size[0]
            right = left + self.crop_size[1]

            crop_image = image[:, top:bottom, left:right]
            crop_mask = mask[:, top:bottom, left:right]

            # クロップマスクのサイズが1000以上の時だけ、image と maskを更新。
            if crop_mask.sum() > 1000:
                resize = Resize()
                image = resize(crop_image)
                mask = resize(crop_mask)
        
        return image, mask


class RandomFlip(object):
    def __init__(self, axis: str, rate=0.5):
        """初期化."""
        self.axis = axis
        self.rate = rate

    def __call__(self, image, mask):
        if np.random.rand() < self.rate:
            if self.axis == "vertical":
                image = image[:, ::-1, :]
                mask = mask[:, ::-1, :]
            elif self.axis == "horizon":
                image = image[:, :, ::-1]
                mask = mask[:, :, ::-1]

        return image, mask


class Move(object):
    def __init__(self, rate=0.5, max_move_rate=0.1, threshold=1000):
        """初期化.

        :argument:
        rate: 移動する確率
        max_move_rate: 最大移動率
        """
        self.rate = rate
        self.max_move_rate = max_move_rate
        self.threshold = threshold

    def __call__(self, image, mask):
        if np.random.rand() < self.rate:
            move_image = image.transpose(1, 2, 0)
            move_mask = mask.transpose(1, 2, 0)

            width = move_image.shape[0]
            height = move_image.shape[1]
            moving_x = width * self.max_move_rate * np.random.random() * np.random.choice([1, -1])
            moving_y = height * self.max_move_rate * np.random.random() * np.random.choice([1, -1])
            move = np.float32([[1, 0, moving_x], [0, 1, moving_y]])

            move_image = cv2.warpAffine(move_image, move, (width, height))
            move_mask = cv2.warpAffine(move_mask, move, (width, height))

            # Update image and mask if mask size is larger than 1000 pix.
            if move_mask.sum() > self.threshold:
                image = move_image.transpose(2, 0, 1)
                mask = move_mask.transpose(2, 0, 1)

        return image, mask


class CutOff(object):
    def __init__(self, mask_size=(20, 50), rate=0.5, threshold=1000):
        self.rate = rate
        self.mask_size = mask_size
        self.threshold = threshold

    def __call__(self, image, mask):
        if np.random.rand() < self.rate:
            cut_image = image.copy()
            cut_mask = mask.copy()

            mask_size = np.random.randint(self.mask_size[0], self.mask_size[1])
            _, h, w = image.shape

            # マスクをかける場所のtop, leftをランダムに決める
            # はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
            top = np.random.randint(0 - mask_size // 2, h - mask_size)
            left = np.random.randint(0 - mask_size // 2, w - mask_size)
            bottom = top + mask_size
            right = left + mask_size

            # はみ出した場合の処理
            if top < 0:
                top = 0
            if left < 0:
                left = 0

            # マスク部分の画素値を0で埋める
            cut_image[:, top:bottom, left:right].fill(0)
            if mask is not None:
                cut_mask[:, top:bottom, left:right].fill(0)

                if cut_mask.sum() > self.threshold:
                    image = cut_image
                    mask = cut_mask
            else:
                mask = None

        return image, mask


class RandomRotation(object):
    def __init__(self, rate, angle_range=(0, 180), threshold=1000):
        self.rate = rate
        self.angle_range = angle_range
        self.threshold = threshold

    def __call__(self, image, mask):
        _, h, w = image.shape
        # resize = Resize((h, w))
        angle = np.random.randint(*self.angle_range)

        t_image = image.transpose(1, 2, 0) # channelを後ろに持ってくる
        t_mask = mask.transpose(1, 2, 0)

        rotate_image = rotate(t_image, angle)
        rotate_mask = rotate(t_mask, angle)
        
        r_h, r_w, _ = rotate_image.shape
        cut_h = int((r_h - h)/2)
        cut_w = int((r_w - w)/2)

        rotate_image = rotate_image[cut_h:cut_h + h, cut_w:cut_w + w, :]
        rotate_mask = rotate_mask[cut_h:cut_h + h, cut_w:cut_w + w, :]
        
        rotate_image = np.where(rotate_image < 0, 0, rotate_image)
        rotate_mask = np.where(rotate_mask < 0, 0, rotate_mask)
        
        if rotate_mask.sum() > self.threshold:
            image = rotate_image.transpose(2, 0, 1)
            mask = rotate_mask.transpose(2, 0, 1)
        return image, mask


class BoundingOnlyDA(object):
    def __init__(self, rate, classes):
        self.rate = rate
        self.classes = classes

    def __call__(self, image, mask):
        # Change Vertival or Horizon to only Bounding Box Only
        if np.random.random() < self.rate:
            try:
                # 0以外のindexをlistで返す
                max_channel_list = np.array([mask[i, :, :].max() for i in range(self.classes)])
                channel_index = np.where(max_channel_list > 0)[0]
                mask_channnel = np.random.choice(channel_index)
                x, y, w, h = cv2.boundingRect(mask[mask_channnel, :, :])

                crop_image = image[:, y:y+h, x:x+w]
                crop_mask = mask[:, y:y+h, x:x+w]

                vertical = RandomFlip(axis='vertical', rate=0.5)
                crop_image, crop_mask = vertical(crop_image, crop_mask)

                horizon = RandomFlip(axis='horizon', rate=0.5)
                crop_image, crop_mask = horizon(crop_image, crop_mask)

                image[:, y:y+h, x:x+w] = crop_image
                mask[:, y:y+h, x:x+w] = crop_mask

            except ValueError:
                pass     

        return image, mask


class Resize(object):
    def __init__(self, resize_size=(256, 256)):
        self.resize_size = resize_size

    def __call__(self, image):
        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, self.resize_size,
                           interpolation=cv2.INTER_AREA)

        image = image.transpose(2, 0, 1)
        return image


class Adjust_gamma(object):
    def __init__(self, rate=0.5, gamma_range=(0.6, 1.4)):
        self.rate = rate
        self.gamma_range = gamma_range

    def __call__(self, image):
        if np.random.rand() < self.rate:
            random_num = np.random.rand() 
            multi_num = self.gamma_range[1] - self.gamma_range[0]
            gamma = random_num * multi_num + self.gamma_range[0]
            image = skimage.exposure.adjust_gamma(image, gamma=gamma)

        return image


class Equalize(object):
    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, image):
        if np.random.rand() < self.rate:
            image = skimage.exposure.equalize_hist(image)

        return image
