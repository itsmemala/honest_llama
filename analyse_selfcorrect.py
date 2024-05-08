import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import json
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt
import argparse
from utils import LogisticRegression_Torch, tokenized_from_file

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def get_probe_wgts(fold,model,results_file_name,save_path,args):
    act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128}
    using_act = 'ah' if '_ah_' in results_file_name else 'mlp_l1' if '_mlp_l1_' in results_file_name else 'mlp'
    num_layers = 32 if '_7B_' in results_file_name else 40 if '_13B_' in results_file_name else 60
    layer = args.custom_layers[model] if args.custom_layers is not None else model if using_act in ['mlp','layer'] else np.floor(model/num_layers) # 0 to 31 -> 0, 32 to 63 -> 1, etc.
    head = 0 if using_act in ['mlp','layer'] else (model%num_layers)
    use_bias = False if 'no_bias' in results_file_name else True
    current_linear_model = LogisticRegression_Torch(act_dims[using_act], 2, use_bias)
    kld_probe = 0
    sim_file_name = results_file_name.replace('individual_linear','individual_linear_unitnorm') if 'unitnorm' not in results_file_name else results_file_name
    try:
        linear_model = torch.load(f'{save_path}/probes/models/{sim_file_name}_model{fold}_{layer}_{head}_{kld_probe}')
    except FileNotFoundError:
        linear_model = torch.load(f'{save_path}/probes/models/{sim_file_name}_model{fold}_{layer}_{head}')
    return linear_model.linear.weight[0], linear_model.linear.weight[1]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy_responses_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--sc_responses_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--sc_responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    greedy_labels = []
    with open(f'{args.save_path}/responses/{args.greedy_responses_labels_file_name}.json', 'r') as read_file:
        for line in read_file:
            greedy_labels.append(json.loads(line))
    sc_labels = []
    with open(f'{args.save_path}/responses/{args.sc_responses_labels_file_name}.json', 'r') as read_file:
        for line in read_file:
            sc_labels.append(json.loads(line))
    sc_responses = []
    with open(f'{args.save_path}/responses/{args.sc_responses_file_name}.json', 'r') as read_file:
        for line in read_file:
            sc_responses.append(json.loads(line))
    
    
    correct_to_incorrect = 0
    incorrect_to_correct = 0
    correct_to_none = 0
    incorrect_to_none = 0
    remains_correct = 0
    remains_incorrect = 0
    sc_labels_val = []
    for idx,row in enumerate(greedy_labels):
        if row['rouge1_to_target']>0.3 and sc_responses[idx]['response1']=="":
            correct_to_none += 1
        elif row['rouge1_to_target']>0.3 and sc_labels[idx]['rouge1_to_target']<=0.3:
            correct_to_incorrect += 1
        elif row['rouge1_to_target']>0.3 and sc_labels[idx]['rouge1_to_target']>0.3:
            remains_correct += 1
        elif row['rouge1_to_target']<=0.3 and sc_responses[idx]['response1']=="":
            incorrect_to_none += 1
        elif row['rouge1_to_target']<=0.3 and sc_labels[idx]['rouge1_to_target']<=0.3:
            remains_incorrect += 1
        elif row['rouge1_to_target']<=0.3 and sc_labels[idx]['rouge1_to_target']>0.3:
            incorrect_to_correct += 1
        sc_label = 1 if sc_labels[idx]['rouge1_to_target']>0.3 else 0
        sc_labels_val.append(sc_label)
    
    print('Remains correct:',remains_correct/len(greedy_labels))
    print('Correct to none:',correct_to_none/len(greedy_labels))
    print('Incorrect to correct:',incorrect_to_correct/len(greedy_labels))
    print('Remains incorrect:',remains_incorrect/len(greedy_labels))
    print('Incorrect to none:',incorrect_to_none/len(greedy_labels))
    print('Correct to incorrect:',correct_to_incorrect/len(greedy_labels))
    
    print('\n')
    print('Total correct:',sum(sc_labels_val))
    
if __name__ == '__main__':
    main()