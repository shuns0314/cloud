# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
import warnings

import joblib
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
import torch

from dataset import Dataset
import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
from utils import str2bool, count_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default=None,
                        help='model name')
    args = parser.parse_args()
    return args


def main():
    val_args = parse_args()
    args = joblib.load(f'models/{val_args.name}/args.pkl')

    if not os.path.exists(f'output/{args.name}'):
        os.makedirs(f'output/{args.name}')

    print('Config -----')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('------------')

    joblib.dump(args, f'models/{args.name}/args.pkl')

    # create model
    print(f"=> creating model {args.arch}")
    model = archs.__dict__[args.arch](args)
    model = model.cuda()

    # Data loading code
    test_img_paths = glob(f'input/{args.dataset}/images/*')
    test_mask_paths = glob(f'input/{args.dataset}/masks/*')

    model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
    model.eval()

    test_dataset = Dataset(img_paths=test_img_paths, 
                           mask_paths=test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (inputs, target) in enumerate(tqdm(test_loader)):
                inputs = inputs.cuda()
                target = target.cuda()

                # compute output
                if args.deepsupervision:
                    output = model(inputs)[-1]
                else:
                    output = model(inputs)

                output = torch.sigmoid(output).data.cpu().numpy()
                img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]

                for i in range(output.shape[0]):
                    imsave('output/%s/'%args.name+os.path.basename(img_paths[i]), (output[i,0,:,:]).astype('uint8'))

        torch.cuda.empty_cache()

    # IoU
    ious = []
    for i in tqdm(range(len(val_mask_paths))):
        mask = imread(val_mask_paths[i])
        pb = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]))

        mask = mask.astype('float32')
        pb = pb.astype('float32') / 255

        iou = iou_score(pb, mask)
        ious.append(iou)
    print('IoU: %.4f' %np.mean(ious))


if __name__ == '__main__':
    main()
