import cv2
import numpy as np
import torch
import torch.utils.data
from scipy.ndimage.interpolation import rotate



class RandomCrop(object):
    """Random Crop."""
    def __init__(self, crop_size=(128, 128), rate=0.5):
        self.crop_size = crop_size
        self.rate = rate

    def __call__(self, image):
        if np.random.rand() < self.rate:
            _, width, height = image.shape
            left = np.random.randint(0, width - self.crop_size[1])
            top = np.random.randint(0, height - self.crop_size[0])

            bottom = top + self.crop_size[0]
            right = left + self.crop_size[1]

            image = image[:, left:right, top:bottom]
            resize = Resize()
            image = resize(image)
        return image


class HorizontalFlip(object):
    def __call__(self, image):
        image = image[:, :, ::-1]
        return image


class VerticalFlip(object):
    def __call__(self, image):
        image = image[:, ::-1, :]
        return image


class CutOff(object):
    def __init__(self, rate=0.5):
        self.rate = rate
        
    def __call__(self, image_origin, mask_size):
        if np.random.rand() < self.rate:
            # 最後に使うfill()は元の画像を書き換えるので、コピーしておく
            image = np.copy(image_origin)
            mask_value = image.mean()

            h, w, _ = image.shape
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

            # マスク部分の画素値を平均値で埋める
            image[:, top:bottom, left:right].fill(mask_value)

        return image


class RandomRotation(object):
    def __init__(self, rate):
        self.rate = rate

    def random_rotation(self, image, angle_range=(0, 180)):
        _, h, w = image.shape
        angle = np.random.randint(*angle_range)
        image = rotate(image, angle)
        resize = Resize((h, w))
        image = resize(image)
        return image

# ----------------------------------------------------------------------------
class Resize(object):
    def __init__(self, resize_size=(256, 256)):
        self.resize_size = resize_size

    def __call__(self, image):
        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, self.resize_size,
                           interpolation=cv2.INTER_AREA)

        image = image.transpose(2, 0, 1)
        return image