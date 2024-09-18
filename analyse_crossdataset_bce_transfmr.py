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
from utils import LogisticRegression_Torch, tokenized_from_file

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

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
    parser.add_argument("--probes_file_name_concat", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--lr_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument("--best_threshold", type=bool, default=False, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    device = 0

    if args.probes_file_name is None: args.probes_file_name = 'hallu_pos'
    hallu_cls = 1 if 'hallu_pos' in args.probes_file_name else 0

    responses, labels = [], []
    if args.responses_file_name!='' and 'baseline' in args.responses_file_name:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                responses.append(data['model_answer'][i])
                if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][i]==True else 0 # pos class is non-hallu
                if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][i]==True else 1 # pos class is hallu
                labels.append(label)
        resp_start_idxs = np.load(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_response_start_token_idx.npy')
    if args.mitigated_responses_file_name!='':
        m_responses, m_labels = [], []
        samples_neg_affected, samples_pos_affected = [], []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                m_responses.append(data['model_answer'][i])
                if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][i]==True else 0 # pos class is non-hallu
                if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][i]==True else 1 # pos class is hallu
                m_labels.append(label)
                if labels[i]!=hallu_cls and label==hallu_cls: samples_neg_affected.append(i)
                if labels[i]==hallu_cls and label!=hallu_cls: samples_pos_affected.append(i)
        print('Num of samples negatively affected:',len(samples_neg_affected))
        print('Num of samples positively affected:',len(samples_pos_affected))
    
    # args.using_act = 'layer' if 'layer' in args.probes_file_name else 'mlp'
    num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0

    if args.dataset_name=='strqa':
        acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        acts_per_file = 20
    else:
        acts_per_file = 100
    
    # print('\nGetting probe predictions on generated responses...')
    # try:
    #     if 'greedy' in args.probes_file_name and ('baseline' in args.responses_file_name or 'dola' in args.responses_file_name):
    #         # Do not use test split results of greedy responses when probes and test file are mismatched
    #         raise FileNotFoundError
    #     else:
    #         # all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_pred.npy')[0]
    #         # labels = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_true.npy')[0][0]
    #         # all_logits = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_logits.npy')[0]
    #         # print(all_preds.shape)
    #         pass
    # except:
    #     try:
    #         all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}.npy')
    #     except FileNotFoundError:
    #         all_preds = []
    #         # Get predictions from probes trained on greedy responses
    #         for layer in range(num_layers):
    #             # Load model
    #             act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
    #             bias = False if 'no_bias' in args.probes_file_name else True
    #             head = 0
    #             kld_probe = 0
    #             try:
    #                 linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}_{kld_probe}')
    #             except FileNotFoundError:
    #                 linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}')
    #             linear_model.eval()
    #             # Load activations
    #             acts = []
    #             for i in range(len(responses)):
    #                 act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
    #                 file_end = i-(i%acts_per_file)+acts_per_file # 487: 487-(87)+100
    #                 file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
    #                 act = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%acts_per_file][layer][head*128:(head*128)+128]).to(device)
    #                 acts.append(act)
    #             inputs = torch.stack(acts,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
    #             if 'unitnorm' in args.probes_file_name or 'individual_linear_orthogonal' in args.probes_file_name or 'individual_linear_specialised' in args.probes_file_name: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
    #             preds = torch.sigmoid(linear_model(inputs).data) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
    #             all_preds.append(preds.cpu().numpy())
    #         all_preds = np.stack(all_preds)
    #         np.save(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}.npy',all_preds)


    def results_at_best_lr(model):
        if args.lr_list is not None:
            probes_file_name_list, auc_by_lr = [], []
            for lr in args.lr_list:
                probes_file_name = args.probes_file_name + str(lr) + '_False' + args.probes_file_name_concat
                probes_file_name_list.append(probes_file_name)
                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{probes_file_name}_val_true.npy', allow_pickle=True).item()
                auc_by_lr.append(roc_auc_score(all_val_true[0][model], np.squeeze(all_val_pred[0][model])))
            best_probes_file_name = probes_file_name_list[np.argmax(auc_by_lr)]
        else:
            best_probes_file_name = args.probes_file_name
        
        all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_true.npy', allow_pickle=True).item()
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
        return best_probes_file_name, all_val_pred, all_val_true, best_t

    # all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_true.npy', allow_pickle=True).item()
    fold = 0
    test_f1_cls0, test_f1_cls1, test_recall_cls0, test_recall_cls1, val_f1_cls1, val_f1_cls0, val_f1_avg = [], [], [], [], [], [], []
    best_probes_per_model, layer_pred_thresholds = [], []
    excl_layers, incl_layers = [], []
    aupr_by_layer, auroc_by_layer = [], []
    num_models = 1 # 33 if args.using_act=='layer' else 32 if args.using_act=='mlp' else 32*32
    print(num_models)
    all_preds = []
    for model in tqdm(range(num_models)):
        best_probes_file_name, all_val_pred, all_val_true, best_t = results_at_best_lr(model)
        best_probes_per_model.append(best_probes_file_name)
        layer_pred_thresholds.append(best_t)
        test_preds = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_pred.npy')[0]
        labels = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_true.npy')[0][0] ## Since labels are same for all models
        all_preds.append(test_preds[model])

        val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
        val_pred_model[val_pred_model>best_t] = 1
        val_pred_model[val_pred_model<=best_t] = 0
        cls1_f1 = f1_score(all_val_true[fold][model],val_pred_model)
        cls0_f1 = f1_score(all_val_true[fold][model],val_pred_model,pos_label=0)
        val_f1_cls0.append(cls0_f1)
        val_f1_cls1.append(cls1_f1)
        val_f1_avg.append(np.mean((cls1_f1,cls0_f1)))
        if cls0_f1==0 or cls1_f1==0:
            excl_layers.append(model)
        else:
            incl_layers.append(model)
        
        test_pred_model = deepcopy(test_preds[model]) # Deep copy so as to not touch orig values
        test_pred_model[test_pred_model>best_t] = 1
        test_pred_model[test_pred_model<=best_t] = 0
        cls1_f1, cls1_re = f1_score(labels,test_pred_model), recall_score(labels,test_pred_model)
        cls0_f1, cls0_re = f1_score(labels,test_pred_model,pos_label=0), recall_score(labels,test_pred_model,pos_label=0)
        test_f1_cls0.append(cls0_f1)
        test_f1_cls1.append(cls1_f1)
        test_recall_cls0.append(cls0_re)
        test_recall_cls1.append(cls1_re)
        precision, recall, _ = precision_recall_curve(labels, np.squeeze(test_preds[model,:,:]))
        aupr_by_layer.append(auc(recall,precision))
        auroc_by_layer.append(roc_auc_score(labels, np.squeeze(test_preds[model,:,:])))
    # print('\nValidation performance:\n',val_f1_avg)
    incl_layers = np.array(incl_layers)
    print('\nExcluded layers:',excl_layers)
    # print(incl_layers)
    # if 'hallu_pos' in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls0),np.mean(test_f1_cls1),'\n') # NH, H
    # if 'hallu_pos' not in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls1),np.mean(test_f1_cls0),'\n') # NH, H
    # if 'hallu_pos' in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls0),np.mean(test_recall_cls1),'\n') # NH, H
    # if 'hallu_pos' not in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls1),np.mean(test_recall_cls0),'\n') # NH, H
    print(np.mean([np.mean(test_f1_cls0),np.mean(test_f1_cls1)]))
    print(np.mean(test_recall_cls1)) # H
    print(np.mean(aupr_by_layer)) # 'Avg AUPR:',
    print(np.mean(auroc_by_layer)) # 'Avg AUROC:',
    # print(auroc_by_layer)
    all_preds = np.stack(all_preds, axis=0)

if __name__ == '__main__':
    main()