import argparse

import matplotlib.pyplot as plt
import pandas as pd


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_curve(path):
    loss_df = pd.read_csv(path)
    loss_df = loss_df.iloc[:, 2:]
    loss_df.plot()
    plt.grid()
    plt.show()

