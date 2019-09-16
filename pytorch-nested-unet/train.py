"""Main module"""
# -*- coding: utf-8 -*-
import os
import argparse
from glob import glob
from collections import OrderedDict

import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from dataset import Dataset
import archs
import metrics
import losses
from utils import str2bool, count_params
from data_augmentation import *


arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
metric_names = list(metrics.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a',
                        metavar='ARCH', default='NestedUNet', choices=arch_names,
                        help='model architecture: '+' | '.join(arch_names)+' (default: NestedUNet)')
    parser.add_argument('--deepsupervision',
                        default=False, type=str2bool)
    parser.add_argument('--img_dataset',
                        default=None, help='image dataset name')
    parser.add_argument('--msk_dataset',
                        default=None, help='mask dataset name')                 
    parser.add_argument('--input-channels',
                        default=3, type=int, help='input channels')
    parser.add_argument('--aug',
                        default=False, type=bool,
                        help='data augmentation')
    parser.add_argument('--loss',
                        default='BCEDiceLoss', choices=loss_names,
                        help='loss: ' + ' | '.join(loss_names) + ' (default: BCEDiceLoss)')
    parser.add_argument('--metric',
                        default='Dice_coef', choices=metric_names,
                        help='metric: ' + ' | '.join(metric_names) + ' (default: Dice_coef)')
    parser.add_argument('--epochs',
                        default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop',
                        default=10, type=int,
                        metavar='N', help='early stopping (default: 10)')
    parser.add_argument('-b', '--batch-size',
                        default=16, type=int,
                        metavar='N', help='mini-batch size (default: 12)')
    parser.add_argument('--optimizer',
                        default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate',
                        default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay',
                        default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov',
                        default=False, type=str2bool, help='nesterov')

    args = parser.parse_args()
    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer):
    avg_losses = AverageMeter()
    avg_metrics = AverageMeter()
    metric_criterion = metrics.__dict__[args.metric]().cuda()
    model.train()

    for inputs, target in tqdm(train_loader):
        inputs = inputs.cuda()
        target = target.cuda()

        # compute output
        if args.deepsupervision:
            outputs = model(inputs)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            metric = metric_criterion(outputs[-1], target)
        else:
            output = model(inputs)
            loss = criterion(output, target)
            metric = metric_criterion(output, target)

        avg_losses.update(loss.item(), inputs.size(0))
        avg_metrics.update(metric, inputs.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', avg_losses.avg),
        ('metric', avg_metrics.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    avg_losses = AverageMeter()
    avg_metrics = AverageMeter()

    # switch to evaluate mode
    model.eval()
    metric_criterion = metrics.__dict__[args.metric]().cuda()
    with torch.no_grad():
        for inputs, target in tqdm(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                metric = metric_criterion(outputs[-1], target)
            else:
                output = model(inputs)
                loss = criterion(output, target)
                metric = metric_criterion(output, target)

            avg_losses.update(loss.item(), inputs.size(0))
            avg_metrics.update(metric, inputs.size(0))

    log = OrderedDict([
        ('loss', avg_losses.avg),
        ('metric', avg_metrics.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.img_dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.img_dataset, args.arch)

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob('inputs/' + args.img_dataset + '/*')
    mask_paths = glob('inputs/' + args.msk_dataset + '/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model = model.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args=args,
                            img_paths=train_img_paths,
                            mask_paths=train_mask_paths,
                            train=True)

    val_dataset = Dataset(args=args,
                          img_paths=val_img_paths,
                          mask_paths=val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'metric', 'val_loss', 'val_metric'
    ])

    best_metric = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - metric %.4f - val_loss %.4f - val_metric %.4f'
              %(train_log['loss'], train_log['metric'], val_log['loss'], val_log['metric']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['metric'],
            val_log['loss'],
            val_log['metric'],
        ], index=['epoch', 'lr', 'loss', 'metric', 'val_loss', 'val_metric'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['metric'] > best_metric:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_metric = val_log['metric']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
