import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import json
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score, classification_report
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt
import seaborn as sns
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
    parser.add_argument("--responses_file_name", type=str, default='', help='local directory with dataset')
    parser.add_argument("--mitigated_responses_file_name", type=str, default='', help='local directory with dataset')
    parser.add_argument("--probes_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--best_threshold", type=bool, default=False, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 0

    if args.probes_file_name is None: args.probes_file_name = 'hallu_pos'
    hallu_cls = 1 if 'hallu_pos' in args.probes_file_name else 0

    responses, labels = [], []
    if args.responses_file_name!='':
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                responses.append(data['model_completion'][i])
                if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][i]==True else 0 # pos class is non-hallu
                if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][i]==True else 1 # pos class is hallu
                labels.append(label)
    if args.mitigated_responses_file_name!='':
        m_responses, m_labels = [], []
        samples_neg_affected, samples_pos_affected = [], []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                m_responses.append(data['model_completion'][i])
                if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][i]==True else 0 # pos class is non-hallu
                if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][i]==True else 1 # pos class is hallu
                m_labels.append(label)
                if labels[i]!=hallu_cls and label==hallu_cls: samples_neg_affected.append(i)
                if labels[i]==hallu_cls and label!=hallu_cls: samples_pos_affected.append(i)
        print('Num of samples negatively affected:',len(samples_neg_affected))
        print('Num of samples positively affected:',len(samples_pos_affected))
    
    num_layers = 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0

    if args.dataset_name=='strqa':
        acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        acts_per_file = 20
    else:
        acts_per_file = 100
    
    print('\nGetting probe predictions on generated responses...')
    try:
        if 'greedy' in args.probes_file_name and ('baseline' in args.responses_file_name or 'dola' in args.responses_file_name):
            # Do not use test split results of greedy responses when probes and test file are mismatched
            raise FileNotFoundError
        else:
            all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_pred.npy')[0]
            labels = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_true.npy')[0][0]
            all_logits = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_logits.npy')[0]
            print(all_preds.shape)
    except:
        try:
            all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}.npy')
        except FileNotFoundError:
            all_preds = []
            # Get predictions from probes trained on greedy responses
            args.using_act = 'layer' if 'layer' in args.probes_file_name else 'mlp'
            for layer in range(num_layers):
                # Load model
                act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
                bias = False if 'no_bias' in args.probes_file_name else True
                head = 0
                kld_probe = 0
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
                if 'unitnorm' in args.probes_file_name or 'individual_linear_orthogonal' in args.probes_file_name or 'individual_linear_specialised' in args.probes_file_name: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                preds = torch.sigmoid(linear_model(inputs).data) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                all_preds.append(preds.cpu().numpy())
            all_preds = np.stack(all_preds)
            np.save(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}.npy',all_preds)
    # Get preds on all tokens
    try:
        alltokens_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}_alltokens_preds.npy')
    except FileNotFoundError:
        alltokens_preds = []
        # Get predictions from probes trained on greedy/baseline responses
        args.using_act = 'layer' if 'layer' in args.probes_file_name else 'mlp'
        args.token = 'all'
        for layer in range(num_layers):
            # Load model
            act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
            bias = False if 'no_bias' in args.probes_file_name else True
            head = 0
            kld_probe = 0
            try:
                linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}_{kld_probe}')
            except FileNotFoundError:
                linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}')
            linear_model.eval()
            # Load activations
            acts = []
            # print(samples_neg_affected[:10] + samples_pos_affected[:10])
            for i in samples_neg_affected[:10] + samples_pos_affected[:10]:
                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                file_end = i-(i%acts_per_file)+acts_per_file # 487: 487-(87)+100
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                inputs = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else None # torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%acts_per_file][layer][head*128:(head*128)+128]).to(device)
                if 'unitnorm' in args.probes_file_name or 'individual_linear_orthogonal' in args.probes_file_name or 'individual_linear_specialised' in args.probes_file_name or ('individual_linear' in args.probes_file_name and 'no_bias' in args.probes_file_name):
                    inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                preds = torch.sigmoid(linear_model(inputs).data)
                alltokens_preds.append(preds.cpu().numpy())
        # np.save(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}_alltokens_preds.npy',alltokens_preds)


    all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_true.npy')
    fold = 0
    test_f1_cls0, test_f1_cls1, val_f1_cls1, val_f1_cls0, val_f1_avg = [], [], [], [], []
    layer_pred_thresholds = []
    for model in range(all_val_pred[fold].shape[0]):
        if args.best_threshold:
            best_val_perf, best_t = 0, 0.5
            for t in [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
                val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
                val_pred_model[val_pred_model>t] = 1
                val_pred_model[val_pred_model<=t] = 0
                cls1_f1 = f1_score(all_val_true[fold][0],val_pred_model)
                cls0_f1 = f1_score(all_val_true[fold][0],val_pred_model,pos_label=0)
                perf = np.mean((cls1_f1,cls0_f1))
                if perf>best_val_perf:
                    best_val_perf, best_t = perf, t
        else:
            best_t = 0.5
        layer_pred_thresholds.append(best_t)
        val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
        val_pred_model[val_pred_model>best_t] = 1
        val_pred_model[val_pred_model<=best_t] = 0
        cls1_f1 = f1_score(all_val_true[fold][0],val_pred_model)
        cls0_f1 = f1_score(all_val_true[fold][0],val_pred_model,pos_label=0)
        val_f1_cls0.append(cls0_f1)
        val_f1_cls1.append(cls1_f1)
        val_f1_avg.append(np.mean((cls1_f1,cls0_f1)))
        test_pred_model = deepcopy(all_preds[model]) # Deep copy so as to not touch orig values
        test_pred_model[test_pred_model>best_t] = 1
        test_pred_model[test_pred_model<=best_t] = 0
        cls1_f1 = f1_score(labels,test_pred_model)
        cls0_f1 = f1_score(labels,test_pred_model,pos_label=0)
        test_f1_cls0.append(cls0_f1)
        test_f1_cls1.append(cls1_f1)
    # print('\nValidation performance:\n',val_f1_avg)
    if 'hallu_pos' in args.probes_file_name: print('\nAverage:',np.mean(test_f1_cls0),np.mean(test_f1_cls1),'\n') # NH, H
    if 'hallu_pos' not in args.probes_file_name: print('\nAverage:',np.mean(test_f1_cls1),np.mean(test_f1_cls0),'\n') # NH, H

    print('\n')
    if len(labels)>0:
        print('\nValidating probe performance...')
        # Last layer probe
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[num_layers-1,i,:])
            confident_sample_pred.append(1 if sample_pred>layer_pred_thresholds[num_layers-1] else 0)
        # print('Using final layer probe:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using final layer probe:\n',classification_report(labels,confident_sample_pred))

        # Best probe from validation data
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[np.argmax(val_f1_avg),i,:])
            confident_sample_pred.append(1 if sample_pred>layer_pred_thresholds[np.argmax(val_f1_avg)] else 0)
        # print('Using best layer probe:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using best layer probe:\n',classification_report(labels,confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            # print(sample_pred.shape,sample_pred[:5])
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        # print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using most confident probe per sample (0.5 threshold):\n',classification_report(labels,confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            # print(sample_pred.shape,sample_pred[:5])
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            layer = np.argmin(probe_wise_entropy)
            confident_sample_pred.append(1 if sample_pred[layer][1]>layer_pred_thresholds[layer] else 0)
        # print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using most confident probe per sample (best val threshold):\n',classification_report(labels,confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            confident_sample_pred.append(1 if np.max(sample_pred)>layer_pred_thresholds[np.argmax(sample_pred)] else 0)
        # print('Using max prob probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using max prob probe per sample (actual prob):\n',classification_report(labels,confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred_dist = [sample_pred[layer]-layer_pred_thresholds[layer] for layer,pred in enumerate(sample_pred)]
            layer = np.argmax(sample_pred_dist)
            confident_sample_pred.append(1 if sample_pred[layer]>layer_pred_thresholds[layer] else 0)
        # print('Using max prob probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Using max prob probe per sample (distance from best val threshold):\n',classification_report(labels,confident_sample_pred))

        # Probe selection - d
        confident_sample_pred = []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred_val = [1 for layer,pred in enumerate(sample_pred) if pred>layer_pred_thresholds[layer]]
            class_1_vote_cnt = sum(sample_pred_val)
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        # print('Voting amongst all probes per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('Voting amongst all probes per sample:\n',classification_report(labels,confident_sample_pred))

        # MC5 Statistics
        confident_sample_pred = []
        mc5_entropy_hallu, mc5_entropy_nonhallu = [], []
        mc5_entropy_hallu_mis, mc5_entropy_nonhallu_mis = 0, 0
        mc5_conf_gap_hallu, mc5_conf_gap_nonhallu = [], []
        for i in range(all_preds.shape[1]):
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred_cls = np.array([1 if pred>layer_pred_thresholds[layer] else 0 for layer,pred in enumerate(sample_pred)])
            sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            best_probe_idxs = np.argpartition(probe_wise_entropy, 5)[:5] # Note: sort is asc, so take first x values for smallest x
            
            # top_5_lower_bound_val = np.max(probe_wise_entropy)
            # best_probe_idxs = probe_wise_entropy<=top_5_lower_bound_val
            # sample_pred_chosen = sample_pred_cls[best_probe_idxs]
            # cls1_vote = np.sum(sample_pred_chosen)/len(sample_pred_chosen)
            # vote_distri = np.array([cls1_vote, 1 - cls1_vote])
            # mc5_entropy = (-vote_distri*np.nan_to_num(np.log2(vote_distri),neginf=0)).sum()
            
            sample_pred_chosen = sample_pred[best_probe_idxs][1]
            # sample_pred_chosen = np.squeeze(all_logits[:,i,:])[best_probe_idxs]*100
            # sample_pred_chosen[sample_pred_chosen<0] = 0
            sample_pred_chosen = np.exp(sample_pred_chosen)/sum(np.exp(sample_pred_chosen))
            mc5_entropy = (-sample_pred_chosen*np.nan_to_num(np.emath.logn(5, sample_pred_chosen),neginf=0)).sum()
            if labels[i]==hallu_cls: mc5_entropy_hallu.append(mc5_entropy)
            if labels[i]!=hallu_cls: mc5_entropy_nonhallu.append(mc5_entropy)
            
            # maj_vote = 1 if cls1_vote>0.5 else 0
            # if labels[i]==hallu_cls and maj_vote!=hallu_cls: mc5_entropy_hallu_mis += 1
            # if labels[i]!=hallu_cls and maj_vote==hallu_cls: mc5_entropy_nonhallu_mis += 1
            # probe_wise_conf = []
            # hallu_vote = cls1_vote if 'hallu_pos' in args.probes_file_name else 1-cls1_vote
            # if hallu_vote>0:
            #     for layer in best_probe_idxs:
            #         probe_wise_conf.append((sample_pred[layer]-layer_pred_thresholds[layer])/(1-layer_pred_thresholds[layer]))
            #     mc5_conf_gap_hallu = np.max(probe_wise_conf) - np.min(probe_wise_conf)
            # if mc5_entropy
            #     confident_sample_pred.append()
        # print('Using entropy among most confident 5 probes:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
        print('MC5 entropy for hallucinations:\n',np.histogram(mc5_entropy_hallu))
        # print('Low entropy and mis-classified as non-hallucination:',mc5_entropy_hallu_mis)
        print('MC5 entropy for non-hallucinations:\n',np.histogram(mc5_entropy_nonhallu))
        # print('Low entropy and mis-classified as hallucination:',mc5_entropy_nonhallu_mis)
        fig, axs = plt.subplots(1,2)
        counts_confident_nh, bins = np.histogram(mc5_entropy_nonhallu, bins=10)
        axs[0].stairs(counts_confident_nh, bins)
        axs[0].title.set_text('Non-Hallucinated')
        counts_confident_nh, bins = np.histogram(mc5_entropy_hallu, bins=10)
        axs[1].stairs(counts_confident_nh, bins)
        axs[1].title.set_text('Hallucinated')
        fig.savefig(f'{args.save_path}/plot1.png')
    
    print('\n')


    # Visualise probe prediction pattern
    for i,sample_preds in enumerate(alltokens_preds):
        sns_fig = sns.heatmap(sample_preds, linewidth=0.5)
        sns_fig.fig.savefig(f'{args.save_path}/predplot{i}.png')


    # Find most confident layers
    # print('\nMost confident layers for hallu...')
    # top_x = 5
    # mc_layers = []
    # min_mc_layers = []
    # min_mc_layers_entropy = []
    # for i in range(all_preds.shape[1]):
    #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
    #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
    #     top_layers = np.argpartition(probe_wise_entropy, top_x)[:top_x]
    #     top_layers_hallu = []
    #     for layer in top_layers:
    #         if np.argmax(sample_pred[layer])==hallu_cls: top_layers_hallu.append(layer)
    #     top_layers_hallu = np.array(top_layers_hallu)
    #     mc_layers.append(top_layers_hallu)
    #     if len(top_layers_hallu)>0: min_mc_layers.append(np.min(top_layers_hallu))
    #     if len(top_layers_hallu)>0: min_mc_layers_entropy.append(np.min(probe_wise_entropy[top_layers_hallu]))
    # # mc_layers = np.array(mc_layers)
    # print(np.histogram(min_mc_layers, bins=range(33)))
    # print(np.histogram(min_mc_layers_entropy, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

    # np.save(f'{args.save_path}/responses/best_layers/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_mc_layers.npy', mc_layers)


    if args.mitigated_responses_file_name!='':
        print('\n\nOriginal perf:',sum(labels)/len(labels) if hallu_cls==0 else 1-(sum(labels)/len(labels)))

        # Self-correct using last layer pred
        final_labels = []
        for i,row in enumerate(labels):
            # Get prediction on orig response
            sample_pred = np.squeeze(all_preds[-1,i,:]) # Get predictions of each sample at last layer
            orig_response_pred = 1 if sample_pred>layer_pred_thresholds[-1] else 0
            if orig_response_pred!=hallu_cls:
                final_labels.append(labels[i])
            else:
                final_labels.append(m_labels[i])
        print('\nDola after using last layer:',sum(final_labels)/len(final_labels) if hallu_cls==0 else 1-(sum(final_labels)/len(final_labels)))
        
        # Self-correct using most confident pred
        final_labels = []
        for i,row in enumerate(labels):
            # Get prediction on orig response
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            orig_response_pred = np.argmax(sample_pred[np.argmin(probe_wise_entropy)])
            if orig_response_pred!=hallu_cls:
                final_labels.append(labels[i])
            else:
                final_labels.append(m_labels[i])
        print('\nDola after using most confident:',sum(final_labels)/len(final_labels) if hallu_cls==0 else 1-(sum(final_labels)/len(final_labels)))
        
        # Self-correct using majority voting pred
        final_labels = []
        for i,row in enumerate(labels):
            # Get prediction on orig response
            sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred_val = [1 for layer,pred in enumerate(sample_pred) if pred>layer_pred_thresholds[layer]]
            class_1_vote_cnt = sum(sample_pred_val)
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            orig_response_pred = maj_vote
            if orig_response_pred!=hallu_cls:
                final_labels.append(labels[i])
            else:
                final_labels.append(m_labels[i])
        print('\nDola after using majority voting:',sum(final_labels)/len(final_labels) if hallu_cls==0 else 1-(sum(final_labels)/len(final_labels)))

if __name__ == '__main__':
    main()