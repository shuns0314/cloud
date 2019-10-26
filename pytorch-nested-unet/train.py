"""Main module"""
# -*- coding: utf-8 -*-
import os
import argparse
from glob import glob
from collections import OrderedDict

import joblib
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
                        help='model architecture: '+' | '.join(arch_names) +' (default: NestedUNet)')
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
                        default=30, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('-b', '--batch-size',
                        default=8, type=int,
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
    parser.add_argument('--n_classes',
                        default=4, type=int, help='n_classes')
    parser.add_argument('--scheduler',
                        default='None', type=str, help='scheduler')
    parser.add_argument('--stratify',
                        default='balance', type=str, help='stratifing method of train_test_split')
    parser.add_argument('--fpa',
                        default=False, type=str2bool)
                        

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


def train(args, train_loader, model, criterion, optimizer, scheduler):
    avg_losses = AverageMeter()
    # avg_metrics = AverageMeter()
    if scheduler is not None:
            scheduler.step()

    # metric_criterion = metrics.__dict__[args.metric]().cuda()
    model.train()

    for inputs, target in tqdm(train_loader):
        inputs = inputs.float().cuda()
        target = target.float().cuda()

        # compute output
        if args.deepsupervision:
            outputs = model(inputs)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            # metric, _ = metric_criterion(outputs[-1], target, best_params=bestparams)
        else:
            output = model(inputs)
            loss = criterion(output, target)
            # metric, _ = metric_criterion(output, target, best_params=bestparams)

        avg_losses.update(loss.item(), inputs.size(0))
        # avg_metrics.update(metric, inputs.size(0))

        # compute gradient and do optimizing step    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', avg_losses.avg),
        ('lr', scheduler.get_lr()),
    ])
    return log


def predict_parameter(args, train_loader, model):
    avg_metrics = AverageMeter()
    avg_threshold_0 = AverageMeter()
    avg_threshold_1 = AverageMeter()
    avg_threshold_2 = AverageMeter()
    avg_threshold_3 = AverageMeter()
    avg_minsize_0 = AverageMeter()
    avg_minsize_1 = AverageMeter()
    avg_minsize_2 = AverageMeter()
    avg_minsize_3 = AverageMeter()
    avg_mask_threshold = AverageMeter()

    metric_criterion = metrics.__dict__[args.metric]().cuda()
    model.eval()
    with torch.no_grad():
        for inputs, target in tqdm(train_loader):
            inputs = inputs.float().cuda()
            target = target.float().cuda()
            # compute output
            output = model(inputs)

            metric, best_params = metric_criterion(output, target, bayes=True)

            avg_metrics.update(metric, inputs.size(0))
            avg_threshold_0.update(best_params[0], inputs.size(0))
            avg_threshold_1.update(best_params[1], inputs.size(0))
            avg_threshold_2.update(best_params[2], inputs.size(0))
            avg_threshold_3.update(best_params[3], inputs.size(0))
            avg_minsize_0.update(best_params[4], inputs.size(0))
            avg_minsize_1.update(best_params[5], inputs.size(0))
            avg_minsize_2.update(best_params[6], inputs.size(0))
            avg_minsize_3.update(best_params[7], inputs.size(0))
            avg_mask_threshold.update(best_params[8], inputs.size(0))

    log = OrderedDict([
        ('metric', avg_metrics.avg),
    ])

    best_params = OrderedDict([
        ('threshold_0', avg_threshold_0.avg),
        ('threshold_1', avg_threshold_1.avg),
        ('threshold_2', avg_threshold_2.avg),
        ('threshold_3', avg_threshold_3.avg),
        ('min_size_0', avg_minsize_0.avg),
        ('min_size_1', avg_minsize_1.avg),
        ('min_size_2', avg_minsize_2.avg),
        ('min_size_3', avg_minsize_3.avg),
        ('mask_threshold', avg_mask_threshold.avg),
    ])

    return log, best_params


def validate(args, val_loader, model, criterion, best_params):
    avg_losses = AverageMeter()
    avg_metrics = AverageMeter()

    # switch to evaluate mode
    model.eval()
    metric_criterion = metrics.__dict__[args.metric]().cuda()
    with torch.no_grad():
        for inputs, target in tqdm(val_loader):
            inputs = inputs.float().cuda()
            target = target.float().cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                if best_params is None:
                    metric = metrics.dice_coef(outputs[-1], target)
                else:
                    metric, _ = metric_criterion(outputs[-1], target, best_params=best_params)

            else:
                outputs = model(inputs)
                loss = criterion(outputs, target)
                if best_params is None:
                    metric = metrics.dice_coef(outputs, target)
                else:
                    metric, _ = metric_criterion(outputs, target, best_params=best_params)

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
            args.name = f'{args.img_dataset}_{args.arch}_wDS'
        else:
            args.name = f'{args.img_dataset}_{args.arch}_woDS'

    if not os.path.exists(f'models/{args.name}' ):
        os.makedirs(f'models/{args.name}')

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
    img_paths = glob(f'inputs/{args.img_dataset}/*')
    mask_paths = glob(f'inputs/{args.msk_dataset}/*')

    # train_test_split
    # stratify split
    if args.stratify is None:
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
            train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

        train_img_paths, parameter_img_paths, train_mask_paths, parameter_mask_paths = \
            train_test_split(train_img_paths, train_mask_paths, test_size=0.1, random_state=1)

    else:
        climate_df = pd.read_csv("../data/20190924_climate.csv", index_col=0)
        if args.stratify == 'sum':
            stratify = climate_df.sum(axis=1).values
        elif args.stratify == 'balance':
            stratify = climate_df.values
            # print(stratify)

        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths, train_str, _ = \
            train_test_split(img_paths, mask_paths, stratify,
                            test_size=0.1, random_state=41, stratify=stratify)

        train_img_paths, parameter_img_paths, train_mask_paths, parameter_mask_paths = \
            train_test_split(train_img_paths, train_mask_paths,
                            test_size=0.1, random_state=1, stratify=train_str)

    # create model
    print(f"=> creating model {args.arch}")
    model = archs.__dict__[args.arch](args)

    model = model.cuda()
    print(count_params(model))

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)

    # Scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
    elif args.scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)
    elif args.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.scheduler == "None":
        scheduler = None

    train_dataset = Dataset(args=args,
                            img_paths=train_img_paths,
                            mask_paths=train_mask_paths,
                            train=True)
    parameter_dataset = Dataset(args=args,
                                img_paths=parameter_img_paths,
                                mask_paths=parameter_mask_paths)
    val_dataset = Dataset(args=args,
                          img_paths=val_img_paths,
                          mask_paths=val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    parameter_loader = torch.utils.data.DataLoader(
        parameter_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'val_loss', 'par_metric', 'val_metric'
    ])

    best_metric = 0
    trigger = 0
    best_params = None

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, scheduler)
        # determine best params
        parameter_log, best_params = predict_parameter(args, parameter_loader, model)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion, best_params)

        print(f"train loss: {train_log['loss']},\
                parmeter metric: {parameter_log['metric']},\
                validation loss: {val_log['loss']},\
                validation metric: {val_log['metric']}")

        tmp = pd.Series([
            epoch, train_log['lr'], train_log['loss'], parameter_log['metric'], val_log['loss'], val_log['metric'],
        ], index=['epoch', 'lr', 'loss', 'val_loss', 'par_metric', 'val_metric'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv(f'models/{args.name}/log.csv', index=False)

        trigger += 1

        if val_log['metric'] > best_metric:
            torch.save(model.state_dict(), f'models/{args.name}/model.pth')
            best_metric = val_log['metric']
            # save best params
            best_params_series = pd.Series(best_params)
            best_params_series.to_csv(f'models/{args.name}/best_params.csv')
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()

    # model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))

    # ここから、パラメタ決定とか
    # for epoch in range(1):
        # parameter_log, best_params = predict_parameter(args, parameter_loader, model)
        # evaluate on validation set
        # val_log = validate(args, val_loader, model, criterion, best_params)

        # print(f"parameter metric: {parameter_log['metric']},\
        #         validation metric: {val_log['metric']}")

        # save best params
        # best_params_series = pd.Series(best_params)
        # best_params_series.to_csv(f'models/{args.name}/best_params.csv')


if __name__ == '__main__':
    main()
