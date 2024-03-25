import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    all_train_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_train_loss.npy')
    all_test_f1s = np.load(f'{args.save_path}/probes/{args.results_file_name}_test_f1.npy')
    for fold in range(len(all_test_f1s)):
        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold]))
        # print('Best:',)
        for model in range(len(all_train_loss[fold])):
            print('Train loss:',all_train_loss[fold][model])


if __name__ == '__main__':
    main()