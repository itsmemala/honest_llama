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
    parser.add_argument("--responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    if args.responses_file_name is not None:
        file_path = f'{args.save_path}/responses/{args.responses_file_name}.json'
        prompts, _, _, _ = tokenized_from_file(file_path, tokenizer)
        catg = {}
        for i in range(4):
            catg[i] = []
        for idx,prompt in enumerate(prompts):
            if 'who' in prompt: catg[0].append(idx)
            if 'when' in prompt: catg[1].append(idx)
            if 'where' in prompt: catg[2].append(idx)
            if 'what' in prompt or 'which' in prompt: catg[3].append(idx)

    try:
        all_val_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_loss.npy')
        all_train_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_train_loss.npy')
    except ValueError:
        all_val_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_loss.npy',allow_pickle=True).item()
        all_train_loss = np.load(f'{args.save_path}/probes/{args.results_file_name}_train_loss.npy',allow_pickle=True).item()
    all_test_f1s = np.load(f'{args.save_path}/probes/{args.results_file_name}_test_f1.npy')
    all_val_f1s = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_f1.npy')
    all_test_pred, all_test_true = np.load(f'{args.save_path}/probes/{args.results_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{args.results_file_name}_test_true.npy')
    all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{args.results_file_name}_val_true.npy')
    for fold in range(len(all_test_f1s)):
        assert sum(all_test_true[fold][0]==all_test_true[fold][1])==len(all_test_true[fold][0]) # check all models have same batch order
        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold]))
        print('Best:',all_test_f1s[fold][np.argmax(all_val_f1s[fold])],np.argmax(all_val_f1s[fold]))
        print('\n')
        best_sample_pred =[]
        num_correct_probes = []
        num_correct_probes_hallu = []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.argmax(sample_pred,axis=1)
            assert sample_pred.shape==(32,) # num_layers
            correct_answer = all_test_true[fold][0][i]
            num_correct_probes.append(sum(sample_pred==correct_answer))
            if correct_answer==0: num_correct_probes_hallu.append(sum(sample_pred==correct_answer))
            # if i==0: print(sample_pred==correct_answer,sum(sample_pred==correct_answer))
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        assert f1_score(all_test_true[fold][0],all_test_true[fold][0])==1
        counts, bins = np.histogram(num_correct_probes)
        plt.stairs(counts, bins)
        plt.savefig(f'{args.save_path}/figures/{args.results_file_name}_oracle_hist.png')
        counts, bins = np.histogram(num_correct_probes_hallu)
        plt.stairs(counts, bins)
        plt.savefig(f'{args.save_path}/figures/{args.results_file_name}_oracle_hist_hallu.png')
        print('Oracle:',f1_score(all_test_true[fold][0],best_sample_pred))
        print('\n')
        confident_sample_pred = []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred))
        # baseline_f1 = f1_score(all_test_true[fold][0],[1 for i in all_test_true[fold][0]])        
        # best_probes = np.argwhere(all_val_f1s[fold]>baseline_f1)
        # print('Baseline:',baseline_f1,'Num of probes > baseline:',len(best_probes))
        best_probes = np.argwhere(all_val_f1s[fold]>=np.mean(all_val_f1s[fold]))
        print('Num of probes > avg:',len(best_probes))
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)[best_probes]
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample (best probes by f1):',f1_score(all_test_true[fold][0],confident_sample_pred))
        # print(all_val_loss[fold])
        best_val_loss_by_model = [np.min(model_losses) for model_losses in all_val_loss[fold]]
        best_probes = np.argwhere(best_val_loss_by_model<=np.mean(best_val_loss_by_model))
        print('Num of probes < avg:',len(best_probes))
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)[best_probes]
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample (best probes by loss):',f1_score(all_test_true[fold][0],confident_sample_pred))
        print('\n')
        np.set_printoptions(precision=2)
        for model in range(len(all_val_loss[fold])):
            print('Val loss model',model,':',all_val_loss[fold][model],'Val F1:',"{:.2f}".format(all_val_f1s[fold][model]),'Test F1:',"{:.2f}".format(all_test_f1s[fold][model]))
        print('\n')
        print('Val and Test f1 correlation across probes:',np.corrcoef(all_val_f1s[fold],all_test_f1s[fold])[0][1])
        print('\n')



if __name__ == '__main__':
    main()