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

    all_val_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_loss.npy')
    all_train_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_train_loss.npy')
    all_test_f1s = np.load(f'{args.save_path}/probes/{args.results_file_name}_test_f1.npy')
    all_val_f1s = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_f1.npy')
    all_test_pred, all_test_true = np.load(f'{args.save_path}/probes/{args.results_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{args.results_file_name}_test_true.npy')
    all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{args.results_file_name}_val_true.npy')
    for fold in range(len(all_test_f1s)):
        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold]))
        print('Best:',all_test_f1s[fold][np.argmax(all_val_f1s[fold])],np.argmax(all_val_f1s[fold]))
        confident_sample_pred = []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred))
        confident_sample_pred = []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)[:27]
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample (excl. 28-32):',f1_score(all_test_true[fold][0],confident_sample_pred))
        # print('\n')
        # for model in range(len(all_val_loss[fold])):
        #     print('Val loss model',model,':',all_val_loss[fold][model])
        # print('\n')


if __name__ == '__main__':
    main()