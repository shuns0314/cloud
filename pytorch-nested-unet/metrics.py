from typing import Tuple, List
import os

from joblib import Parallel, delayed
from multiprocessing import Pool
from multiprocessing import Process
import numpy as np
import torch
from torch import nn
import cv2
from GPyOpt.methods import BayesianOptimization

def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:, 0, :, :]
    target = target[:, 0, :, :]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)



class Dice_coef(nn.Module):
    def __init__(self, output=None, target=None, bayes=True):
        super(Dice_coef, self).__init__()
        self.output = output
        self.target = target
        self.bayes = bayes

    def forward(self, output, target, bayes=True, best_params=None):
        self.output = output
        self.target = target
        self.bayes = bayes

        # vest params is None when it is first train.
        if best_params is None:
            var_args = [
                0.5, 0.5, 0.5, 0.5,  # threshold
                0.0, 0.0, 0.0, 0.0,  # minsizes
                0.2,
            ]
        else:
            var_args = [
                best_params['threshold_0'],
                best_params['threshold_1'],
                best_params['threshold_2'],
                best_params['threshold_3'],
                best_params['min_size_0'],
                best_params['min_size_1'],
                best_params['min_size_2'],
                best_params['min_size_3'],
                best_params['mask_threshold'],
            ]

        if bayes is True:
            pbounds = [
                {'name': 'threshold_0', 'type': 'continuous', 'domain': (0, 1.0)},
                {'name': 'threshold_1', 'type': 'continuous', 'domain': (0, 1.0)},
                {'name': 'threshold_2', 'type': 'continuous', 'domain': (0, 1.0)},
                {'name': 'threshold_3', 'type': 'continuous', 'domain': (0, 1.0)},
                {'name': 'min_size_0', 'type': 'continuous', 'domain': (100, 10000)},
                {'name': 'min_size_1', 'type': 'continuous', 'domain': (100, 10000)},
                {'name': 'min_size_2', 'type': 'continuous', 'domain': (100, 10000)},
                {'name': 'min_size_3', 'type': 'continuous', 'domain': (100, 10000)},
                {'name': 'mask_threshold', 'type': 'continuous', 'domain': (0, 1.0)},
                ]

            optimizer = BayesianOptimization(
                f=self.calc_loss,
                domain=pbounds,
                num_cores=16,
                maximize=True,
                verbosity=False,
                initial_design_numdata=2
                )

            optimizer.run_optimization(max_iter=18, verbosity=False)

            dice_coefficient = -1 * (optimizer.fx_opt)
            best_params = optimizer.x_opt

        else:
            dice_coefficient = self.calc_loss(var_args)

        return dice_coefficient, best_params

    def post_process(self, mask, threshold: int, min_size: Tuple[int, int], batch, channel):
        """小さい面積のmaskは消す.

        :arguments:
        mask: [h, w, channel]
        """
        mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros(mask.shape, np.float32)

        num = 0
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1
                num += 1
        return (predictions, batch, channel)

    def channel_post_process(self, 
                             mask: np.ndarray, 
                             thresholds: List[float], 
                             min_sizes: List[int], 
                             mask_threshold):
        """全チャンネルに対して、post_processを実行する.

        :arguments:
        mask: [channel, h, w]
        """
        mask = mask.detach().cpu().numpy()
        pred_mask = np.zeros(mask.shape)

        for batch in range(mask.shape[0]):
            for channel in range(mask.shape[1]):
                # 閾値より大きな値を１、それ以下を0にする。また、予測領域が小さい場合は消す。
                mask_p, _, _ = self.post_process(mask[batch, channel, :, :],
                                                 thresholds[channel],
                                                 min_sizes[channel],
                                                 batch, channel)
                # 予測領域の形状を１にする。
                _, contours, _ = cv2.findContours(mask_p.astype('uint8'), 1, 2)
                for cont in contours:
                    x, y, w, h = cv2.boundingRect(cont)
                    pred_mask[batch, channel, y:y+h, x:x+w] = 1

        # 矩形に囲った場所でも、予測値の低い箇所は0にする。
        pred_mask = np.where(mask < mask_threshold, 0, pred_mask)

        return pred_mask

    def calc_loss(self, x):
        """Detemine thresholds and minsize of mask by Baysian Optimize.

        minsizeは画像サイズで変えたほうがよいかも。
        """
        smooth = 1e-5
        thresholds = [x[:, 0], x[:, 1], x[:, 2], x[:, 3]]
        min_sizes = [x[:, 4], x[:, 5], x[:, 6], x[:, 7]]
        mask_threshold = x[:, 8]

        output = self.channel_post_process(torch.sigmoid(self.output), thresholds, min_sizes, mask_threshold)

        output = output.reshape(-1)
        target = self.target.view(-1).data.cpu().numpy()

        intersection = (output * target).sum()
        dice_coefficient = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
        return dice_coefficient


def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)
