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
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument("--greedy_responses_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--sc_responses_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--greedy_responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--sc_responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--greedy_results_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 0

    greedy_labels = []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.greedy_responses_labels_file_name}.json', 'r') as read_file:
        for line in read_file:
            greedy_labels.append(json.loads(line))
    greedy_responses = []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.greedy_responses_file_name}.json', 'r') as read_file:
        for line in read_file:
            greedy_responses.append(json.loads(line))
    sc_labels = []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.sc_responses_labels_file_name}.json', 'r') as read_file:
        for line in read_file:
            sc_labels.append(json.loads(line))
    sc_responses = []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.sc_responses_file_name}.json', 'r') as read_file:
        for line in read_file:
            sc_responses.append(json.loads(line))
    
    
    correct_to_incorrect = []
    incorrect_to_correct = []
    correct_to_none = 0
    incorrect_to_none = 0
    remains_correct = []
    remains_incorrect = []
    sc_labels_val, combined_labels = [], []
    is_different = 0
    for idx,row in enumerate(greedy_labels):
        if row['rouge1_to_target']>0.3 and sc_responses[idx]['response1']=="":
            correct_to_none += 1
        elif row['rouge1_to_target']>0.3 and sc_labels[idx]['rouge1_to_target']<=0.3:
            correct_to_incorrect.append(idx)
        elif row['rouge1_to_target']>0.3 and sc_labels[idx]['rouge1_to_target']>0.3:
            remains_correct.append(idx)
        elif row['rouge1_to_target']<=0.3 and sc_responses[idx]['response1']=="":
            incorrect_to_none += 1
        elif row['rouge1_to_target']<=0.3 and sc_labels[idx]['rouge1_to_target']<=0.3:
            remains_incorrect.append(idx)
        elif row['rouge1_to_target']<=0.3 and sc_labels[idx]['rouge1_to_target']>0.3:
            incorrect_to_correct.append(idx)
        
        if sc_responses[idx]['response1'] != "" and sc_responses[idx]['response1'] != greedy_responses[idx]['response1']:
            is_different += 1
        
        sc_label = 1 if sc_labels[idx]['rouge1_to_target']>0.3 else 0
        sc_labels_val.append(sc_label)
    
    print('Remains correct:',len(remains_correct)*100/len(greedy_labels))
    print('Correct to none:',correct_to_none*100/len(greedy_labels))
    print('Incorrect to correct:',len(incorrect_to_correct)*100/len(greedy_labels))
    print('Remains incorrect:',len(remains_incorrect)*100/len(greedy_labels))
    print('Incorrect to none:',incorrect_to_none*100/len(greedy_labels))
    print('Correct to incorrect:',len(correct_to_incorrect)*100/len(greedy_labels))
    
    print('\n')
    print('Total correct:',sum(sc_labels_val)*100/len(greedy_labels))
    print('Total different:',is_different*100/len(greedy_labels))

    all_test_pred, all_test_true = np.load(f'{args.save_path}/probes/{args.greedy_results_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{args.greedy_results_file_name}_test_true.npy')
    all_test_pred, all_test_true = all_test_pred[0], all_test_true[0] # fold-0
    
    # print('\nQualitative Analysis:')
    # for idx in remains_correct:
    #     print(idx, greedy_labels[idx]['rouge1_to_target'], sc_labels[idx]['rouge1_to_target'])
    #     print(sc_responses[idx])

    print('\nGetting probe predictions on selfcorrect responses...')
    all_sc_preds = []
    # Get predictions from probes trained on greedy responses
    try:
        all_sc_preds = np.load(f'{args.save_path}/probes/{args.greedy_results_file_name}_{args.sc_responses_file_name}.npy')
    except FileNotFoundError:
        num_layers = 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0
        for layer in range(num_layers):
            # Load model
            act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
            bias = False if 'no_bias' in args.greedy_results_file_name else True
            head = 0
            kld_probe = 0
            linear_model = LogisticRegression_Torch(act_dims[args.using_act], 2, bias=bias).to(device)
            try:
                linear_model = torch.load(f'{args.save_path}/probes/models/{args.greedy_results_file_name}_model0_{layer}_{head}_{kld_probe}')
            except FileNotFoundError:
                linear_model = torch.load(f'{args.save_path}/probes/models/{args.greedy_results_file_name}_model0_{layer}_{head}')
            linear_model.eval()
            # Load activations
            acts = []
            for i in range(len(sc_labels)):
                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                file_end = i-(i%100)+100 # 487: 487-(87)+100
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.sc_responses_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
                acts.append(act)
            inputs = torch.stack(acts,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
            if 'unitnorm' in args.greedy_results_file_name: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            sc_preds = F.softmax(linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
            all_sc_preds.append(sc_preds.cpu().numpy())
        all_sc_preds = np.stack(all_sc_preds)
        np.save(f'{args.save_path}/probes/{args.greedy_results_file_name}_{args.sc_responses_file_name}.npy',all_sc_preds)

    print('\n')
    display_score = 'acc' # f1
    sc_labels_val = np.array(sc_labels_val)
    for subset_num,resp_subset in enumerate([remains_correct, incorrect_to_correct, remains_incorrect, correct_to_incorrect]):
        print(subset_num+1)

        print('On original responses:')

        # Baseline - last layer prediction
        confident_sample_pred = []
        print(all_test_pred.shape)
        for i in resp_subset:
            sample_pred = np.squeeze(all_test_pred[-1,i,:]) # Get predictions of each sample at last layer
            confident_sample_pred.append(np.argmax(sample_pred))
        if display_score=='f1':
            print('Using last layer:',f1_score(all_test_true[0][resp_subset],confident_sample_pred),f1_score(all_test_true[0][resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Using last layer:',accuracy_score(all_test_true[0][resp_subset],confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in resp_subset:
            sample_pred = np.squeeze(all_test_pred[:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        if display_score=='f1':
            print('Using most confident probe per sample:',f1_score(all_test_true[0][resp_subset],confident_sample_pred),f1_score(all_test_true[0][resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Using most confident probe per sample:',accuracy_score(all_test_true[0][resp_subset],confident_sample_pred))

        # Probe selection - d
        confident_sample_pred = []
        for i in resp_subset:
            sample_pred = np.squeeze(all_test_pred[:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        if display_score=='f1':
            print('Voting amongst all probes per sample:',f1_score(all_test_true[0][resp_subset],confident_sample_pred),f1_score(all_test_true[0][resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Voting amongst all probes per sample:',accuracy_score(all_test_true[0][resp_subset],confident_sample_pred))

        print('On self-correct responses:')

        # Baseline - last layer prediction
        confident_sample_pred = []
        print(all_sc_preds.shape)
        for i in resp_subset:
            sample_pred = np.squeeze(all_sc_preds[-1,i,:]) # Get predictions of each sample at last layer
            confident_sample_pred.append(np.argmax(sample_pred))
        if display_score=='f1':
            print('Using last layer:',f1_score(sc_labels_val[resp_subset],confident_sample_pred),f1_score(sc_labels_val[resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Using last layer:',accuracy_score(sc_labels_val[resp_subset],confident_sample_pred))

        # Probe selection - a
        confident_sample_pred = []
        for i in resp_subset:
            sample_pred = np.squeeze(all_sc_preds[:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        if display_score=='f1':
            print('Using most confident probe per sample:',f1_score(sc_labels_val[resp_subset],confident_sample_pred),f1_score(sc_labels_val[resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Using most confident probe per sample:',accuracy_score(sc_labels_val[resp_subset],confident_sample_pred))

        # Probe selection - d
        confident_sample_pred = []
        for i in resp_subset:
            sample_pred = np.squeeze(all_sc_preds[:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        if display_score=='f1':
            print('Voting amongst all probes per sample:',f1_score(sc_labels_val[resp_subset],confident_sample_pred),f1_score(sc_labels_val[resp_subset],confident_sample_pred,pos_label=0))
        else:
            print('Voting amongst all probes per sample:',accuracy_score(sc_labels_val[resp_subset],confident_sample_pred))
    
    hallu_cls = 1 if 'hallu_pos' in args.greedy_results_file_name else 0

    print('\n\nOriginal perf:',sum(all_test_true[0])/len(all_test_true[0]))

    # Self-correct using last layer pred
    final_labels = []
    for i,row in enumerate(greedy_labels):
        # Get prediction on orig response
        sample_pred = np.squeeze(all_test_pred[-1,i,:]) # Get predictions of each sample at last layer
        orig_response_pred = np.argmax(sample_pred)
        if orig_response_pred!=hallu_cls:
            final_labels.append(all_test_true[0][i])
        else:
            final_labels.append(sc_labels_val[i])
    print('\n\Self-correct using last layer:',sum(final_labels)/len(final_labels))
    
    
    # Self-correct using most confident pred
    final_labels = []
    for idx,row in enumerate(greedy_labels):
        # Get prediction on orig response
        sample_pred = np.squeeze(all_test_pred[:,i,:]) # Get predictions of each sample across all layers of model
        probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
        orig_response_pred = np.argmax(sample_pred[np.argmin(probe_wise_entropy)])
        if orig_response_pred!=hallu_cls:
            final_labels.append(all_test_true[0][i])
        else:
            final_labels.append(sc_labels_val[i])
    print('\n\Self-correct using most confident:',sum(final_labels)/len(final_labels))
    
    # Self-correct using majority voting pred
    final_labels = []
    for idx,row in enumerate(greedy_labels):
        # Get prediction on orig response
        sample_pred = np.squeeze(all_test_pred[:,i,:]) # Get predictions of each sample across all layers of model
        class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
        maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
        orig_response_pred = maj_vote
        if orig_response_pred!=hallu_cls:
            final_labels.append(all_test_true[0][i])
        else:
            final_labels.append(sc_labels_val[i])
    print('\n\Self-correct using majority voting:',sum(final_labels)/len(final_labels))
    
    
if __name__ == '__main__':
    main()