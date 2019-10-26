import argparse

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default=None, help='path')
    args = parser.parse_args()
    return args

def plot_loss_curve(path):
    loss_df = pd.read_csv(path)
    loss_df = loss_df.iloc[:, 2:]
    loss_df.plot()
    plt.grid()
    plt.show()

def main():
    args = parse_args()
    plot_loss_curve(args.path)

if __name__ == '__main__':
    main()
