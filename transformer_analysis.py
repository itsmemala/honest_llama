import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import json
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from utils import My_Transformer_Layer, tokenized_from_file

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('model_name', type=str, default='llama_7B')
    # parser.add_argument('dataset_name', type=str, default='strqa')
    parser.add_argument("--probes_file_name1", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--probes_file_name2", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 0

    # if args.probes_file_name is None: args.probes_file_name = 'hallu_pos'
    # hallu_cls = 1 if 'hallu_pos' in args.probes_file_name else 0
    
    # args.using_act = 'layer' if 'layer' in args.probes_file_name else 'mlp'
    # num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0

    # if args.dataset_name=='strqa':
    #     acts_per_file = 50
    # elif args.dataset_name=='gsm8k':
    #     acts_per_file = 20
    # else:
    #     acts_per_file = 100
    
    all_preds1 = np.squeeze(np.load(f'{args.save_path}/probes/{args.probes_file_name1}_test_pred.npy')[0])
    all_preds2 = np.squeeze(np.load(f'{args.save_path}/probes/{args.probes_file_name2}_test_pred.npy')[0])
    labels = np.squeeze(np.load(f'{args.save_path}/probes/{args.probes_file_name1}_test_true.npy')[0])
    print(all_preds1.shape, all_preds2.shape, labels.shape)


    # all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_true.npy', allow_pickle=True).item()

if __name__ == '__main__':
    main()