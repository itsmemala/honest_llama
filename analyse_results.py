import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from matplotlib import pyplot as plt

def main():

    all_train_loss = np.load(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_train_loss.npy')
    all_test_f1s = np.load(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_test_f1.npy')
    for fold in range(len(all_test_f1s)):
        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold])
        # prkint('Best:')
    

if __name__ == '__main__':
    main()