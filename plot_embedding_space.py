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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from utils import LogisticRegression_Torch, tokenized_from_file

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='strqa')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument("--probes_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 'cuda'

    # Load model
    nlinear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}').to(device)

    if args.dataset_name=='strqa':
        args.acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        args.acts_per_file = 20
    else:
        args.acts_per_file = 100
    
    if 'strqa' in args.test_file_name:
        args.test_acts_per_file = 50
    elif 'gsm8k' in args.test_file_name:
        args.test_acts_per_file = 20
    else:
        args.test_acts_per_file = 100

    # Load acts
    my_train_acts, my_test_acts = [], []
    for idx in train_idxs:
        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
        if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
            # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
            act = combine_acts(idx,args.train_file_name,args)
            if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
            if args.no_sep==False:
                sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                act = torch.cat((act,sep_token), dim=1)
            act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
        else:
            try:
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
            except torch.cuda.OutOfMemoryError:
                device_id += 1
                device = 'cuda:'+str(device_id) # move to next gpu when prev is filled; test data load and rest of the processing can happen on the last gpu
                print('Loading on device',device_id)
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        my_train_acts.append(act)

    # if args.token=='tagged_tokens': my_train_acts = torch.nn.utils.rnn.pad_sequence(my_train_acts, batch_first=True)
    
    if args.test_file_name is not None:
        for idx in test_idxs:
            file_end = idx-(idx%args.test_acts_per_file)+args.test_acts_per_file # 487: 487-(87)+100
            test_dataset_name = args.test_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa')
            file_path = f'{args.save_path}/features/{args.model_name}_{test_dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
            if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file]).to(device)
                act = combine_acts(idx,args.test_file_name,args)
                if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                if args.no_sep==False:
                    sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                    act = torch.cat((act,sep_token), dim=1)
                act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
            else:
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file]).to(device)
            my_test_acts.append(act)
        # if args.token=='tagged_tokens': my_test_acts = torch.nn.utils.rnn.pad_sequence(my_test_acts, batch_first=True)
    my_train_acts, my_test_acts = torch.stack(my_train_acts), torch.stack(my_test_acts)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(my_train_acts)
    print(tsne.kl_divergence_)


if __name__ == '__main__':
    main()