import os
import torch
import torch.nn.functional as F
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

# def get_probe_wgts(fold,model,results_file_name,save_path,args):
#     act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128}
#     using_act = 'ah' if '_ah_' in results_file_name else 'mlp_l1' if '_mlp_l1_' in results_file_name else 'mlp'
#     num_layers = 32 if '_7B_' in results_file_name else 40 if '_13B_' in results_file_name else 60
#     layer = args.custom_layers[model] if args.custom_layers is not None else model if using_act in ['mlp','layer'] else np.floor(model/num_layers) # 0 to 31 -> 0, 32 to 63 -> 1, etc.
#     head = 0 if using_act in ['mlp','layer'] else (model%num_layers)
#     use_bias = False if 'no_bias' in results_file_name else True
#     current_linear_model = LogisticRegression_Torch(act_dims[using_act], 2, use_bias)
#     kld_probe = 0
#     sim_file_name = results_file_name.replace('individual_linear','individual_linear_unitnorm') if 'unitnorm' not in results_file_name else results_file_name
#     try:
#         linear_model = torch.load(f'{save_path}/probes/models/{sim_file_name}_model{fold}_{layer}_{head}_{kld_probe}')
#     except FileNotFoundError:
#         linear_model = torch.load(f'{save_path}/probes/models/{sim_file_name}_model{fold}_{layer}_{head}')
#     return linear_model.linear.weight[0], linear_model.linear.weight[1]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='strqa')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument("--responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--mitigated_responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--probes_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 0

    responses, labels = [], []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}.json', 'r') as read_file:
        data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            responses.append(data['model_completion'][i])
            label = 1 if data['is_correct'][i]==True else 0
            labels.append(label)
    if  args.mitigated_responses_file_name is not None:
        m_responses, m_labels = [], []
        samples_neg_affected = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                m_responses.append(data['model_completion'][i])
                label = 1 if data['is_correct'][i]==True else 0
                m_labels.append(label)
                if labels[i]==1 and label==0: samples_neg_affected.append(i)
        print('Num of samples negatively affected:',len(samples_neg_affected))
    
    if args.dataset_name=='strqa':
        acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        acts_per_file = 20
    else:
        acts_per_file = 100
    
    print('\nGetting probe predictions on generated responses...')
    all_preds = []
    # Get predictions from probes trained on greedy responses
    num_layers = 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0
    for layer in range(num_layers):
        # Load model
        act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
        bias = False if 'no_bias' in args.probes_file_name else True
        head = 0
        kld_probe = 0
        linear_model = LogisticRegression_Torch(act_dims[args.using_act], 2, bias=bias).to(device)
        try:
            linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}_{kld_probe}')
        except FileNotFoundError:
            linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}')
        linear_model.eval()
        # Load activations
        acts = []
        for i in range(len(responses)):
            act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
            file_end = i-(i%acts_per_file)+acts_per_file # 487: 487-(87)+100
            file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
            act = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%acts_per_file][layer][head*128:(head*128)+128]).to(device)
            acts.append(act)
        inputs = torch.stack(acts,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
        if 'unitnorm' in args.probes_file_name: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
        preds = F.softmax(linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
        all_preds.append(preds.cpu().numpy())
    all_preds = np.stack(all_preds)

    print('\n')
    if len(labels)>0:
         # Oracle 2
        best_sample_pred =[]
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            best_probe_idxs = np.argpartition(probe_wise_entropy, -5)[-5:]
            top_5_lower_bound_val = np.min(probe_wise_entropy[best_probe_idxs])
            best_probe_idxs = probe_wise_entropy>=top_5_lower_bound_val
            sample_pred_chosen = sample_pred[best_probe_idxs]
            sample_pred_chosen = np.argmax(sample_pred_chosen,axis=1)
            correct_answer = labels[i]
            # if i==0: print(sample_pred_chosen,sample_pred_chosen==correct_answer)
            if sum(sample_pred_chosen==correct_answer)>0: # If any one is correct
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        print('Oracle (using 5 most confident):',f1_score(labels,best_sample_pred),f1_score(labels,best_sample_pred,pos_label=0))

        print('\nValidating probe performance...')
        # Last layer probe
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[num_layers-1,i,:])
            confident_sample_pred.append(np.argmax(sample_pred))
        print('Using final layer probe:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))

        # Probe selection - a
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))

        # Probe selection - d
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst all probes per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
    # Find most confident layers
    print('\nMost confident layers for hallu...')
    top_x = 5
    mc_layers = []
    min_mc_layers = []
    min_mc_layers_entropy = []
    for i in range(all_preds.shape[1]):
        sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
        probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
        top_layers = np.argpartition(probe_wise_entropy, top_x)[:top_x]
        top_layers_hallu = []
        for layer in top_layers:
            if np.argmax(sample_pred[layer])==0: top_layers_hallu.append(layer)
        top_layers_hallu = np.array(top_layers_hallu)
        mc_layers.append(top_layers_hallu)
        if len(top_layers_hallu)>0: min_mc_layers.append(np.min(top_layers_hallu))
        if len(top_layers_hallu)>0: min_mc_layers_entropy.append(np.min(probe_wise_entropy[top_layers_hallu]))
    # mc_layers = np.array(mc_layers)
    print(np.histogram(min_mc_layers, bins=range(33)))
    print(np.histogram(min_mc_layers_entropy, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

    np.save(f'{args.save_path}/responses/best_layers/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_mc_layers.npy', mc_layers)

if __name__ == '__main__':
    main()