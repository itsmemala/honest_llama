import os
import sys
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
    parser.add_argument("--probes_file_name_concat", type=str, default='', help='local directory with dataset')
    parser.add_argument('--lr_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument('--seed_list',default=None,type=list_of_ints,required=False,help='(default=%(default)s)')
    parser.add_argument("--best_hyp_using_aufpr", type=bool, default=False, help='local directory with dataset')
    parser.add_argument("--best_threshold", type=bool, default=False, help='local directory with dataset')
    parser.add_argument("--best_thr_using_recall", type=bool, default=False, help='local directory with dataset')
    parser.add_argument('--fpr_at_recall',type=float, default=0.95)
    parser.add_argument('--aufpr_from',type=float, default=0.0)
    parser.add_argument('--aufpr_till',type=float, default=1.0)
    parser.add_argument("--min_max_scale_dist", type=bool, default=False, help='')
    parser.add_argument("--best_hyp_on_test", type=bool, default=False, help='')
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
    num_layers = 1 #33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0
    num_models = 1 #33 if args.using_act=='layer' else 32 if args.using_act=='mlp' else 32*32
    if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name): 
        print('\n\nSETTING NUM_LAYERS=1\n\n')
        num_layers, num_models = 1, 1 # We only ran these for the last layer

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
            # all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_pred.npy')[0]
            # labels = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_true.npy')[0][0]
            # all_logits = np.load(f'{args.save_path}/probes/{args.probes_file_name}_test_logits.npy')[0]
            # print(all_preds.shape)
            pass
    except:
        try:
            all_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}.npy')
        except FileNotFoundError:
            all_preds = []
            # Get predictions from probes trained on greedy responses
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


    all_results_list = []

    for seed in args.seed_list:
        args.probes_file_name = 'NLSC'+str(seed)+'_'+args.probes_file_name.split('_',1)[1]
        seed_results_list = []

        # val_pred_model,all_val_true[fold][0]
        def my_aufpr(preds,labels,getfull=False):
            preds, labels = np.squeeze(preds), np.squeeze(labels)
            r_list, fpr_list = [], []
            # thresholds = np.histogram_bin_edges(preds, bins='sqrt') if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else [x / 100.0 for x in range(0, 105, 5)]
            if (('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name)) and args.min_max_scale_dist==False:
                thresholds = np.histogram_bin_edges(preds, bins='sqrt')
            else:
                thresholds = [x / 100.0 for x in range(0, 105, 5)]
            for t in thresholds:
                thr_preds = deepcopy(preds) # Deep copy so as to not touch orig values
                if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                    thr_preds[preds<=t] = 1 # <= to ensure correct classification when dist = [-1,0]
                    thr_preds[preds>t] = 0
                else:
                    thr_preds[preds>t] = 1
                    thr_preds[preds<=t] = 0
                assert thr_preds.shape==labels.shape
                fp = np.sum((thr_preds == 1) & (labels == 0))
                tn = np.sum((thr_preds == 0) & (labels == 0))
                r_list.append(recall_score(labels,thr_preds))
                fpr_list.append(fp / (fp + tn))
            r_list, fpr_list = np.array(r_list), np.array(fpr_list)
            recall_vals, fpr_at_recall_vals = [], []
            check_recall_intervals = [x / 100.0 for x in range(0, 105, 5)]
            for check_recall in check_recall_intervals:
                try: 
                    # fpr_at_recall_vals.append(np.min(fpr_list[np.argwhere(r_list>=check_recall)]))
                    fpr_at_recall_vals.append(np.min(fpr_list[np.argwhere((r_list>=check_recall) & (r_list<check_recall+0.05))]))
                    recall_vals.append(check_recall)
                except ValueError:
                    continue
            # interpolate and fill missing values
            # print(recall_vals,fpr_at_recall_vals)
            fpr_at_recall_vals = np.interp(check_recall_intervals, recall_vals, fpr_at_recall_vals)
            # print(check_recall_intervals,fpr_at_recall_vals)
            # sys.exit()
            if getfull:
                aufpr = auc(check_recall_intervals,fpr_at_recall_vals)
            else:
                check_recall_intervals,fpr_at_recall_vals = np.array(check_recall_intervals), np.array(fpr_at_recall_vals)
                aufpr_idxes = (check_recall_intervals>=args.aufpr_from) & (check_recall_intervals<=args.aufpr_till)
                # print(check_recall_intervals[aufpr_idxes])
                # sys.exit()
                aufpr = auc(check_recall_intervals[aufpr_idxes],fpr_at_recall_vals[aufpr_idxes])
            return check_recall_intervals, fpr_at_recall_vals, aufpr

        def results_at_best_lr(model):
            if args.lr_list is not None:
                probes_file_name_list, perf_by_lr = [], []
                for lr in args.lr_list:
                    probes_file_name = args.probes_file_name + str(lr) + '_False' + args.probes_file_name_concat
                    try:
                        if args.best_hyp_on_test:
                            all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{probes_file_name}_test_true.npy')
                        else:
                            all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{probes_file_name}_val_true.npy', allow_pickle=True).item()
                    except FileNotFoundError:
                        probes_file_name = probes_file_name.replace("/","")
                        if args.best_hyp_on_test:
                            all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{probes_file_name}_test_true.npy')
                        else:
                            all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{probes_file_name}_val_true.npy', allow_pickle=True).item()
                    probes_file_name_list.append(probes_file_name)
                    if args.min_max_scale_dist: all_val_pred[0][model] = (all_val_pred[0][model] - all_val_pred[0][model].min()) / (all_val_pred[0][model].max() - all_val_pred[0][model].min()) # min-max-scale distances
                    try:
                        auc_val = roc_auc_score(all_val_true[0][model], [-v for v in all_val_pred[0][model]]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(all_val_true[0][model], np.squeeze(all_val_pred[0][model]))
                        _, _, aufpr_val = my_aufpr(all_val_pred[0][model],all_val_true[0][model],getfull=True)
                    except ValueError:
                        print('\n\nVALUE ERROR FOR PROBE:',probes_file_name,'\n\n') # for nlinear probe at lr=0.5 and min_max_scale_dist=True
                        auc_val, aufpr_val = 0, 100
                    perf_by_lr.append(aufpr_val if args.best_hyp_using_aufpr else auc_val)
                best_probes_file_name = probes_file_name_list[np.argmin(perf_by_lr)] if args.best_hyp_using_aufpr else probes_file_name_list[np.argmax(perf_by_lr)]
                print(best_probes_file_name)
            else:
                best_probes_file_name = args.probes_file_name
            
            if args.best_hyp_on_test:
                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_true.npy')
            else:
                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_true.npy', allow_pickle=True).item()
            if args.best_threshold:
                if args.best_thr_using_recall:
                    best_val_fpr, best_t = 1, 0
                else:
                    best_val_perf, best_t = 0, 0.5
                val_dist_min, val_dist_max = all_val_pred[fold][model].min(), all_val_pred[fold][model].max()
                if args.min_max_scale_dist: all_val_pred[fold][model] = (all_val_pred[fold][model] - val_dist_min) / (val_dist_max - val_dist_min) # min-max-scale distances
                # thresholds = np.histogram_bin_edges(all_val_pred[fold][model], bins='sqrt') if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else [0,0.05,0.10,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
                if (('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name)) and args.min_max_scale_dist==False:
                    thresholds = np.histogram_bin_edges(all_val_pred[fold][model], bins='sqrt')
                else:
                    thresholds = [x / 100.0 for x in range(0, 105, 5)]
                for t in thresholds:
                    val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
                    if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                        val_pred_model[all_val_pred[fold][model]<=t] = 1
                        val_pred_model[all_val_pred[fold][model]>t] = 0
                    else:
                        val_pred_model[all_val_pred[fold][model]>t] = 1
                        val_pred_model[all_val_pred[fold][model]<=t] = 0
                    cls1_f1 = f1_score(all_val_true[fold][0],val_pred_model)
                    cls0_f1 = f1_score(all_val_true[fold][0],val_pred_model,pos_label=0)
                    recall = recall_score(all_val_true[fold][0],val_pred_model)
                    if args.best_thr_using_recall:
                        fp = np.sum((np.squeeze(val_pred_model) == 1) & (np.squeeze(all_val_true[fold][0]) == 0))
                        tn = np.sum((np.squeeze(val_pred_model) == 0) & (np.squeeze(all_val_true[fold][0]) == 0))
                        val_fpr = fp / (fp + tn)
                        if recall >= 0.9:
                            if val_fpr<best_val_fpr:
                                best_val_fpr, best_t = val_fpr, t
                    else:
                        perf = np.mean((cls1_f1,cls0_f1)) # cls1_f1
                        if perf>best_val_perf:
                            best_val_perf, best_t = perf, t
            else:
                best_t = 0.5
            return best_probes_file_name, all_val_pred, all_val_true, best_t, val_dist_min, val_dist_max

        # all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_pred.npy', allow_pickle=True).item(), np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_true.npy', allow_pickle=True).item()
        fold = 0
        test_f1_cls0, test_f1_cls1, test_recall_cls0, test_recall_cls1, val_f1_cls1, val_f1_cls0, val_f1_avg = [], [], [], [], [], [], []
        best_probes_per_model, layer_pred_thresholds = [], []
        excl_layers, incl_layers = [], []
        aupr_by_layer, auroc_by_layer = [], []
        # num_models = 1 # 33 if args.using_act=='layer' else 32 if args.using_act=='mlp' else 32*32
        print(num_models)
        all_preds = []
        for model in tqdm(range(num_models)):
            best_probes_file_name, all_val_pred, all_val_true, best_t, val_dist_min, val_dist_max = results_at_best_lr(model)
            best_probes_per_model.append(best_probes_file_name)
            layer_pred_thresholds.append(best_t)
            test_preds = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_pred.npy')[0]
            labels = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_true.npy')[0][0] ## Since labels are same for all models
            
            val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
            if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                val_pred_model[all_val_pred[fold][model]<best_t] = 1
                val_pred_model[all_val_pred[fold][model]>=best_t] = 0
            else:
                val_pred_model[all_val_pred[fold][model]>best_t] = 1
                val_pred_model[all_val_pred[fold][model]<=best_t] = 0
            cls1_f1 = f1_score(all_val_true[fold][model],val_pred_model)
            cls0_f1 = f1_score(all_val_true[fold][model],val_pred_model,pos_label=0)
            val_f1_cls0.append(cls0_f1)
            val_f1_cls1.append(cls1_f1)
            val_f1_avg.append(np.mean((cls1_f1,cls0_f1)))
            if cls0_f1==0 or cls1_f1==0:
                excl_layers.append(model)
            else:
                incl_layers.append(model)
            
            if args.min_max_scale_dist: test_preds[model] = (test_preds[model] - val_dist_min) / (val_dist_max - val_dist_min)# min-max-scale distances using val distances
            all_preds.append(test_preds[model])
            test_pred_model = deepcopy(test_preds[model]) # Deep copy so as to not touch orig values
            if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                test_pred_model[test_preds[model]<best_t] = 1
                test_pred_model[test_preds[model]>=best_t] = 0
            else:
                test_pred_model[test_preds[model]>best_t] = 1
                test_pred_model[test_preds[model]<=best_t] = 0
            cls1_f1, cls1_re = f1_score(labels,test_pred_model), recall_score(labels,test_pred_model)
            cls0_f1, cls0_re = f1_score(labels,test_pred_model,pos_label=0), recall_score(labels,test_pred_model,pos_label=0)
            test_f1_cls0.append(cls0_f1)
            test_f1_cls1.append(cls1_f1)
            test_recall_cls0.append(cls0_re)
            test_recall_cls1.append(cls1_re)
            precision, recall, _ = precision_recall_curve(labels, np.squeeze(test_preds[model]))
            aupr_by_layer.append(auc(recall,precision))
            auroc_by_layer.append(roc_auc_score(labels, [-v for v in np.squeeze(test_preds[model])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels, np.squeeze(test_preds[model])))
        # print('\nValidation performance:\n',val_f1_avg)
        incl_layers = np.array(incl_layers)
        print('\nExcluded layers:',excl_layers)
        # print(incl_layers)
        # if 'hallu_pos' in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls0),np.mean(test_f1_cls1),'\n') # NH, H
        # if 'hallu_pos' not in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls1),np.mean(test_f1_cls0),'\n') # NH, H
        # if 'hallu_pos' in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls0),np.mean(test_recall_cls1),'\n') # NH, H
        # if 'hallu_pos' not in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls1),np.mean(test_recall_cls0),'\n') # NH, H
        
        ########################
        # seed_results_list.append(np.mean([np.mean(test_f1_cls0),np.mean(test_f1_cls1)])) # print(np.mean([np.mean(test_f1_cls0),np.mean(test_f1_cls1)]))
        # seed_results_list.append(np.mean(test_recall_cls1)) # print(np.mean(test_recall_cls1)) # H
        # seed_results_list.append(np.mean(aupr_by_layer)) # print(np.mean(aupr_by_layer)) # 'Avg AUPR:',
        # seed_results_list.append(np.mean(auroc_by_layer)) # print(np.mean(auroc_by_layer)) # 'Avg AUROC:',
        ########################

        # print(auroc_by_layer)
        # print(len(all_preds),all_preds[0].shape)
        all_preds = np.stack(all_preds, axis=0)
        # print(all_preds.shape)


        # print('\n')
        if len(labels)>0:
            # print('\nValidating probe performance...')
            # Last layer probe
            confident_sample_pred = []
            for i in range(all_preds.shape[1]):
                sample_pred = np.squeeze(all_preds[num_layers-1,i])
                if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                    confident_sample_pred.append(1 if sample_pred<=layer_pred_thresholds[num_layers-1] else 0)
                else:
                    confident_sample_pred.append(1 if sample_pred>layer_pred_thresholds[num_layers-1] else 0)
            # print('Using final layer probe:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using final layer probe:\n',classification_report(labels,confident_sample_pred))

            confident_sample_pred = np.array(confident_sample_pred)
            fp = np.sum((confident_sample_pred == 1) & (labels == 0))
            tn = np.sum((confident_sample_pred == 0) & (labels == 0))
            test_fpr_best_f1 = fp / (fp + tn)

            # fn = (confident_sample_pred == 0) & (labels == 1)
            # fn_index = np.where(fn)[0]
            # print('# fn:',len(fn_index))
            # print('Index of fn:',fn_index)
            # sys.exit()

            ########################
            # print(layer_pred_thresholds[num_layers-1], fpr_list, r_list)
            seed_results_list.append(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)])) # print(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)]))
            # seed_results_list.append(best_r)
            # seed_results_list.append(test_fpr_best_r)
            if args.fpr_at_recall==-1:
                # Create dirs if does not exist:
                if not os.path.exists(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}'):
                    os.makedirs(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}', exist_ok=True)
                test_preds, model = all_preds, num_layers-1
                recall_vals, fpr_at_recall_vals, aucfpr = my_aufpr(test_preds[model],labels)
                fig, axs = plt.subplots(1,1)
                axs.plot(recall_vals,fpr_at_recall_vals)
                for xy in zip(recall_vals,fpr_at_recall_vals):
                    axs.annotate('(%.2f, %.2f)' % xy, xy=xy)
                axs.set_xlabel('Recall')
                axs.set_ylabel('FPR')
                axs.title.set_text('FPR at recall')
                fig.savefig(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall.png')
                seed_results_list.append(aucfpr)
                np.save(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall_xaxis.npy',np.array(recall_vals))
                np.save(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall_yaxis.npy',np.array(fpr_at_recall_vals))
            else:
                seed_results_list.append(test_fpr)
            seed_results_list.append(test_fpr_best_f1)
            seed_results_list.append(f1_score(labels,confident_sample_pred))
            seed_results_list.append(precision_score(labels,confident_sample_pred))
            seed_results_list.append(recall_score(labels,confident_sample_pred)) # print(recall_score(labels,confident_sample_pred))
            precision, recall, thresholds = precision_recall_curve(labels, np.squeeze(all_preds[num_layers-1]))
            seed_results_list.append(auc(recall,precision)) # print(auc(recall,precision))
            seed_results_list.append(roc_auc_score(labels, [-v for v in np.squeeze(all_preds[num_layers-1])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels,np.squeeze(all_preds[num_layers-1]))) # print(roc_auc_score(labels,np.squeeze(all_preds[num_layers-1,:,:])))
            ########################

            # # Best probe from validation data
            # ma_layer = np.argmax(val_f1_avg)
            # confident_sample_pred = []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[ma_layer,i])
            #     if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
            #         confident_sample_pred.append(1 if sample_pred<=layer_pred_thresholds[ma_layer] else 0)
            #     else:
            #         confident_sample_pred.append(1 if sample_pred>layer_pred_thresholds[ma_layer] else 0)
            # # print('Using best layer probe:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # # print('Using best layer probe:\n',classification_report(labels,confident_sample_pred))
            
            # confident_sample_pred = np.array(confident_sample_pred)
            # fp = np.sum((confident_sample_pred == 1) & (labels == 0))
            # tn = np.sum((confident_sample_pred == 0) & (labels == 0))
            # test_fpr_best_f1 = fp / (fp + tn)

            # #######################
            # seed_results_list.append(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)])) # print(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)]))
            # if args.fpr_at_recall==-1:
            #     recall_vals, fpr_at_recall_vals, aucfpr = my_aufpr(all_preds[ma_layer],labels)
            #     seed_results_list.append(aucfpr)
            # else:
            #     seed_results_list.append(test_fpr)
            # seed_results_list.append(test_fpr_best_f1)
            # seed_results_list.append(f1_score(labels,confident_sample_pred))
            # seed_results_list.append(precision_score(labels,confident_sample_pred))
            # seed_results_list.append(recall_score(labels,confident_sample_pred)) # print(recall_score(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, np.squeeze(all_preds[ma_layer]))
            # seed_results_list.append(auc(recall,precision)) # print(auc(recall,precision))
            # seed_results_list.append(roc_auc_score(labels, [-v for v in np.squeeze(all_preds[ma_layer])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels,np.squeeze(all_preds[ma_layer]))) # print(roc_auc_score(labels,np.squeeze(all_preds[ma_layer])))
            ########################

            #####################################################################################################################################
            # Probe selection - a
            # confident_sample_pred,confident_sample_probs = [], []
            # mc_layer = []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            #     # print(sample_pred.shape,sample_pred[:5])
            #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            #     layer = np.argmin(probe_wise_entropy)
            #     confident_sample_pred.append(np.argmax(sample_pred[layer]))
            #     confident_sample_probs.append(np.squeeze(all_preds[layer,i,:]))
            #     if confident_sample_pred[-1]==hallu_cls: mc_layer.append(layer)
            # # print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using most confident probe per sample (0.5 threshold):\n',classification_report(labels,confident_sample_pred))
            # print('\nMc Layer:\n',np.histogram(mc_layer, bins=range(num_layers+1)))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # print('AUPR:',auc(recall,precision))
            # print('AUROC:',roc_auc_score(labels,confident_sample_probs))

            # Probe selection - a
            # confident_sample_pred = []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            #     # print(sample_pred.shape,sample_pred[:5])
            #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            #     layer = np.argmin(probe_wise_entropy)
            #     confident_sample_pred.append(1 if sample_pred[layer][1]>layer_pred_thresholds[layer] else 0)
            # # print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using most confident probe per sample (best val threshold):\n',classification_report(labels,confident_sample_pred))

            # # Probe selection - a
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i]) # Get predictions of each sample across all layers of model
            #     sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            #     # print(sample_pred.shape,sample_pred[:5])
            #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            #     layer = incl_layers[np.argmin(probe_wise_entropy[incl_layers])]
            #     confident_sample_pred.append(1 if sample_pred[layer][1]>layer_pred_thresholds[layer] else 0)
            #     confident_sample_probs.append(np.squeeze(all_preds[layer,i]))
            # # print('Using most confident probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # # print('Using most confident probe per sample (best val threshold, excl layers):\n',classification_report(labels,confident_sample_pred))
            
            # confident_sample_pred = np.array(confident_sample_pred)
            # fp = np.sum((confident_sample_pred == 1) & (labels == 0))
            # tn = np.sum((confident_sample_pred == 0) & (labels == 0))
            # test_fpr_best_f1 = fp / (fp + tn)

            # #######################
            # seed_results_list.append(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)])) # print(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)]))
            # if args.fpr_at_recall==-1:
            #     recall_vals, fpr_at_recall_vals, aucfpr = my_aufpr(confident_sample_probs,labels)
            #     seed_results_list.append(aucfpr)
            # else:
            #     seed_results_list.append(test_fpr)
            # seed_results_list.append(test_fpr_best_f1)
            # seed_results_list.append(f1_score(labels,confident_sample_pred))
            # seed_results_list.append(precision_score(labels,confident_sample_pred))
            # seed_results_list.append(recall_score(labels,confident_sample_pred)) # print(recall_score(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # seed_results_list.append(auc(recall,precision)) # print(auc(recall,precision))
            # seed_results_list.append(roc_auc_score(labels, [-v for v in np.squeeze(confident_sample_probs)]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels,confident_sample_probs)) # print(roc_auc_score(labels,confident_sample_probs))
            #######################

            #####################################################################################################################################
            # Probe selection - a
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     confident_sample_pred.append(1 if np.max(sample_pred)>layer_pred_thresholds[np.argmax(sample_pred)] else 0) # Using layer with max prob
            #     confident_sample_probs.append(np.max(sample_pred))
            # # print('Using max prob probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using max prob probe per sample (actual prob):\n',classification_report(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # print('AUPR:',auc(recall,precision))
            # print('AUROC:',roc_auc_score(labels,confident_sample_probs))
            # print('AUROC (cls0):',roc_auc_score([not bool(label) for label in labels],[1-prob for prob in confident_sample_probs]))

            
            # best_f1, best_single_threshold, best_report = 0, 0, None
            # for t in thresholds:
            #     confident_sample_pred = []
            #     for i in range(all_preds.shape[1]):
            #         sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #         confident_sample_pred.append(1 if np.max(sample_pred)>t else 0) # Using layer with max prob
            #     if f1_score(labels,confident_sample_pred)>best_f1:
            #         best_f1, best_single_threshold = f1_score(labels,confident_sample_pred,average='macro'), t
            #         best_report = classification_report(labels,confident_sample_pred)
            # print('Results at threshold with best macro-F1:\n',best_report)


            # Probe selection - a
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred_dist = [sample_pred[layer]-layer_pred_thresholds[layer] for layer,pred in enumerate(sample_pred)]  # Using layer with max dist from threshold
            #     layer = np.argmax(sample_pred_dist)
            #     confident_sample_pred.append(1 if sample_pred[layer]>layer_pred_thresholds[layer] else 0)
            #     confident_sample_probs.append(np.squeeze(all_preds[layer,i,:]))
            # # print('Using max prob probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using max prob probe per sample (distance from best val threshold):\n',classification_report(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # print('AUPR:',auc(recall,precision))
            # print('AUROC:',roc_auc_score(labels,confident_sample_probs))


            # best_f1, best_single_threshold, best_report = 0, 0, None
            # for t in thresholds:
            #     confident_sample_pred = []
            #     for i in range(all_preds.shape[1]):
            #         sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #         confident_sample_pred.append(1 if np.max(sample_pred)>t else 0) # Using layer with max prob
            #     if f1_score(labels,confident_sample_pred)>best_f1:
            #         best_f1, best_single_threshold = f1_score(labels,confident_sample_pred,average='macro'), t
            #         best_report = classification_report(labels,confident_sample_pred)
            # print('Results at threshold with best macro-F1:\n',best_report)


            # Probe selection - a
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred_dist = [sample_pred[layer]-layer_pred_thresholds[layer] for layer,pred in enumerate(sample_pred)]
            #     sample_pred_dist = np.array(sample_pred_dist)
            #     layer = incl_layers[np.argmax(sample_pred_dist[incl_layers])]
            #     confident_sample_pred.append(1 if sample_pred[layer]>layer_pred_thresholds[layer] else 0)
            #     confident_sample_probs.append(np.squeeze(all_preds[layer,i,:]))
            # # print('Using max prob probe per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Using max prob probe per sample (distance from best val threshold, excl layers):\n',classification_report(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # print('AUPR:',auc(recall,precision))
            # print('AUROC:',roc_auc_score(labels,confident_sample_probs))
            #####################################################################################################################################

            # Probe selection - d
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred_val = [1 for layer,pred in enumerate(sample_pred) if pred>layer_pred_thresholds[layer]]
            #     class_1_vote_cnt = sum(sample_pred_val)
            #     maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            #     confident_sample_pred.append(maj_vote)
            #     confident_sample_probs.append(class_1_vote_cnt/sample_pred.shape[0])
            # # print('Voting amongst all probes per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('Voting amongst all probes per sample:\n',classification_report(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # print('AUPR:',auc(recall,precision))
            # print('AUROC:',roc_auc_score(labels,confident_sample_probs))

            # # Probe selection - d
            # confident_sample_pred,confident_sample_probs = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i]) # Get predictions of each sample across all layers of model
            #     sample_pred_val = [1 if pred>layer_pred_thresholds[layer] else 0 for layer,pred in enumerate(sample_pred)]
            #     sample_pred_val = np.array(sample_pred_val)
            #     class_1_vote_cnt = sum(sample_pred_val[incl_layers])
            #     maj_vote = 1 if class_1_vote_cnt>=(len(incl_layers)/2) else 0
            #     confident_sample_pred.append(maj_vote)
            #     confident_sample_probs.append(class_1_vote_cnt/sample_pred.shape[0])
            # # print('Voting amongst all probes per sample:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # # print('Voting amongst all probes per sample (excl layers):\n',classification_report(labels,confident_sample_pred))
            
            # confident_sample_pred = np.array(confident_sample_pred)
            # fp = np.sum((confident_sample_pred == 1) & (labels == 0))
            # tn = np.sum((confident_sample_pred == 0) & (labels == 0))
            # test_fpr_best_f1 = fp / (fp + tn)

            # #######################
            # seed_results_list.append(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)])) # print(np.mean([f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0)]))
            # if args.fpr_at_recall==-1:
            #     recall_vals, fpr_at_recall_vals, aucfpr = my_aufpr(confident_sample_probs,labels)
            #     seed_results_list.append(aucfpr)
            # else:
            #     seed_results_list.append(test_fpr)
            # seed_results_list.append(test_fpr_best_f1)
            # seed_results_list.append(f1_score(labels,confident_sample_pred))
            # seed_results_list.append(precision_score(labels,confident_sample_pred))
            # seed_results_list.append(recall_score(labels,confident_sample_pred)) # print(recall_score(labels,confident_sample_pred))
            # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
            # seed_results_list.append(auc(recall,precision)) # print(auc(recall,precision))
            # seed_results_list.append(roc_auc_score(labels, [-v for v in np.squeeze(confident_sample_probs)]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels,confident_sample_probs)) # print(roc_auc_score(labels,confident_sample_probs))
            ########################

            # MC5 Statistics
            # confident_sample_pred = []
            # mc5_entropy_hallu, mc5_entropy_nonhallu = [], []
            # mc5_entropy_hallu_mis, mc5_entropy_nonhallu_mis = 0, 0
            # mc5_conf_gap_hallu, mc5_conf_gap_nonhallu = [], []
            # for i in range(all_preds.shape[1]):
            #     sample_pred = np.squeeze(all_preds[:,i,:]) # Get predictions of each sample across all layers of model
            #     sample_pred_cls = np.array([1 if pred>layer_pred_thresholds[layer] else 0 for layer,pred in enumerate(sample_pred)])
            #     sample_pred = np.concatenate((1-sample_pred[:, None], sample_pred[:, None]),axis=1)
            #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            #     best_probe_idxs = np.argpartition(probe_wise_entropy, 5)[:5] # Note: sort is asc, so take first x values for smallest x
                
            #     # top_5_lower_bound_val = np.max(probe_wise_entropy)
            #     # best_probe_idxs = probe_wise_entropy<=top_5_lower_bound_val
            #     # sample_pred_chosen = sample_pred_cls[best_probe_idxs]
            #     # cls1_vote = np.sum(sample_pred_chosen)/len(sample_pred_chosen)
            #     # vote_distri = np.array([cls1_vote, 1 - cls1_vote])
            #     # mc5_entropy = (-vote_distri*np.nan_to_num(np.log2(vote_distri),neginf=0)).sum()
                
            #     sample_pred_chosen = sample_pred[best_probe_idxs][1]
            #     # sample_pred_chosen = np.squeeze(all_logits[:,i,:])[best_probe_idxs]*100
            #     # sample_pred_chosen[sample_pred_chosen<0] = 0
            #     sample_pred_chosen = np.exp(sample_pred_chosen)/sum(np.exp(sample_pred_chosen))
            #     mc5_entropy = (-sample_pred_chosen*np.nan_to_num(np.emath.logn(5, sample_pred_chosen),neginf=0)).sum()
            #     if labels[i]==hallu_cls: mc5_entropy_hallu.append(mc5_entropy)
            #     if labels[i]!=hallu_cls: mc5_entropy_nonhallu.append(mc5_entropy)
                
            #     # maj_vote = 1 if cls1_vote>0.5 else 0
            #     # if labels[i]==hallu_cls and maj_vote!=hallu_cls: mc5_entropy_hallu_mis += 1
            #     # if labels[i]!=hallu_cls and maj_vote==hallu_cls: mc5_entropy_nonhallu_mis += 1
            #     # probe_wise_conf = []
            #     # hallu_vote = cls1_vote if 'hallu_pos' in args.probes_file_name else 1-cls1_vote
            #     # if hallu_vote>0:
            #     #     for layer in best_probe_idxs:
            #     #         probe_wise_conf.append((sample_pred[layer]-layer_pred_thresholds[layer])/(1-layer_pred_thresholds[layer]))
            #     #     mc5_conf_gap_hallu = np.max(probe_wise_conf) - np.min(probe_wise_conf)
            #     # if mc5_entropy
            #     #     confident_sample_pred.append()
            # # print('Using entropy among most confident 5 probes:',f1_score(labels,confident_sample_pred),f1_score(labels,confident_sample_pred,pos_label=0))
            # print('MC5 entropy for hallucinations:\n',np.histogram(mc5_entropy_hallu))
            # # print('Low entropy and mis-classified as non-hallucination:',mc5_entropy_hallu_mis)
            # print('MC5 entropy for non-hallucinations:\n',np.histogram(mc5_entropy_nonhallu))
            # # print('Low entropy and mis-classified as hallucination:',mc5_entropy_nonhallu_mis)
            # fig, axs = plt.subplots(1,2)
            # counts_confident_nh, bins = np.histogram(mc5_entropy_nonhallu, bins=10)
            # axs[0].stairs(counts_confident_nh, bins)
            # axs[0].title.set_text('Non-Hallucinated')
            # counts_confident_nh, bins = np.histogram(mc5_entropy_hallu, bins=10)
            # axs[1].stairs(counts_confident_nh, bins)
            # axs[1].title.set_text('Hallucinated')
            # fig.savefig(f'{args.save_path}/plot1.png')
        
        print('\n')
        print(np.argmax(val_f1_avg))

        all_results_list.append(np.array(seed_results_list))
    print(np.mean(np.stack(all_results_list)*100,axis=0).tolist())
    print(np.std(np.stack(all_results_list)*100,axis=0).tolist())

    # Get preds on all tokens
    # if args.responses_file_name=='' and args.dataset_name=='tqa_gen':
    #     args.responses_file_name = 'greedy_responses_test'
    #     resp_start_idxs = np.load(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_response_start_token_idx.npy')
    # try:
    #     alltokens_preds = np.load(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}_alltokens_preds.npy', allow_pickle=True)
    #     # raise FileNotFoundError
    # except FileNotFoundError:
    #     resp_start_idxs = np.load(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_response_start_token_idx.npy')
    #     alltokens_preds = []
    #     # Get predictions from probes trained on greedy/baseline responses
    #     args.using_act = 'layer' if 'layer' in args.probes_file_name else 'mlp'
    #     args.token = 'all'
    #     act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
    #     bias = False if 'no_bias' in args.probes_file_name else True
    #     head = 0
    #     kld_probe = 0
    #     # for i in tqdm(samples_neg_affected[:10] + samples_pos_affected[:10]):
    #     for i in tqdm(range(len(labels))):
    #         # Load activations
    #         act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
    #         file_end = i-(i%acts_per_file)+acts_per_file # 487: 487-(87)+100
    #         file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
    #         acts_by_layer = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%acts_per_file]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else None # torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%acts_per_file][layer][head*128:(head*128)+128]).to(device)
    #         preds_by_layer = []
    #         for layer in range(num_layers):
    #             # Load model
    #             try:
    #                 linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}_{kld_probe}')
    #             except FileNotFoundError:
    #                 linear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}_model0_{layer}_{head}')
    #             linear_model.eval()
    #             inputs = acts_by_layer[layer][resp_start_idxs[i]:]
    #             if 'unitnorm' in args.probes_file_name or 'individual_linear_orthogonal' in args.probes_file_name or 'individual_linear_specialised' in args.probes_file_name or ('individual_linear' in args.probes_file_name and 'no_bias' in args.probes_file_name):
    #                 inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
    #             preds = torch.sigmoid(linear_model(inputs).data).cpu().numpy()
    #             preds_by_layer.append(np.array([pred-layer_pred_thresholds[layer] for pred in preds]))
    #         preds_by_layer = np.stack(preds_by_layer)
    #         alltokens_preds.append(np.squeeze(preds_by_layer))
    #         # print(i,responses[i])
    #     alltokens_preds_arr = np.empty(len(alltokens_preds), object)                                                        
    #     alltokens_preds_arr[:] = alltokens_preds
    #     np.save(f'{args.save_path}/probes/{args.probes_file_name}_{args.responses_file_name}_alltokens_preds.npy',alltokens_preds_arr)

    # # Visualise probe prediction pattern
    # for i,sample_preds in tqdm(enumerate(alltokens_preds)):
    #     fig, axs = plt.subplots(1,1)
    #     sns_fig = sns.heatmap(sample_preds, linewidth=0.5)
    #     sns_fig.get_figure().savefig(f'{args.save_path}/predplot{i}.png')

    # print('\nAnalyse probe prediction pattern across all answer tokens...')
    # confident_sample_pred,confident_sample_probs = [], []
    # mc_layer = []
    # mc_layers = [] # Layers to be used for contrast in dola
    # for i,sample_preds in tqdm(enumerate(alltokens_preds)):
    #     agg_layer_preds = []
    #     for layer_preds in sample_preds:
    #         agg_layer_preds.append(np.mean(layer_preds)) # Avg predictions across all tokens at a given layer
    #     agg_layer_preds = np.array(agg_layer_preds)
    #     agg_layer_preds = np.concatenate((1-agg_layer_preds[:, None], agg_layer_preds[:, None]),axis=1)
    #     probe_wise_entropy = (-agg_layer_preds*np.nan_to_num(np.log2(agg_layer_preds),neginf=0)).sum(axis=1)
    #     layer = np.argmin(probe_wise_entropy)
    #     confident_sample_pred.append(1 if agg_layer_preds[layer][1]>0 else 0) # Note this is already the distance from threshold, therefore we check for >0
    #     confident_sample_probs.append(agg_layer_preds[layer][1])
    #     if confident_sample_pred[-1]==hallu_cls: 
    #         mc_layer.append(layer)
    #         mc_layers.append(layer)
    #     else:
    #         mc_layers.append(-1) # If not predicted as hallucination, don't contrast, just use normal decoding
    # print('Averaging across tokens and using most confident probe:\n',classification_report(labels,confident_sample_pred))
    # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
    # print('AUC:',auc(recall,precision))
    # print('\nMc Layer:\n',np.histogram(mc_layer, bins=range(num_layers+1)))
    # np.save(f'{args.save_path}/responses/best_layers/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_mc_layers_tokenavg.npy', mc_layers)

    # confident_sample_pred,confident_sample_probs = [], []
    # mc_layer = []
    # mc_layers = [] # Layers to be used for contrast in dola
    # error_idxs = []
    # no_hallu_preds = 0
    # for i,sample_preds in tqdm(enumerate(alltokens_preds)):
    #     agg_layer_preds = []
    #     for layer_preds in sample_preds:
    #         try:
    #             agg_layer_preds.append(np.max(layer_preds)) # Maxpool predictions across all tokens at a given layer
    #         except ValueError:
    #             agg_layer_preds.append(0)
    #             if i not in error_idxs:
    #                 print(i)
    #                 error_idxs.append(i)
    #     agg_layer_preds = np.array(agg_layer_preds)
    #     agg_layer_preds = np.concatenate((1-agg_layer_preds[:, None], agg_layer_preds[:, None]),axis=1)
    #     probe_wise_entropy = (-agg_layer_preds*np.nan_to_num(np.log2(agg_layer_preds),neginf=0)).sum(axis=1)
    #     layer = np.argmin(probe_wise_entropy)
    #     confident_sample_pred.append(1 if agg_layer_preds[layer][1]>0 else 0) # Note this is already the distance from threshold, therefore we check for >0
    #     confident_sample_probs.append(agg_layer_preds[layer][1])
    #     if confident_sample_pred[-1]==hallu_cls: 
    #         mc_layer.append(layer)
    #         mc_layers.append(layer)
    #     else:
    #         # mc_layers.append(-1) # If not predicted as hallucination, don't contrast, just use normal decoding
    #         layers_predicting = []
    #          # Find all layers where prediction is hallu
    #         for layer in range(agg_layer_preds.shape[0]):
    #             pred = 1 if agg_layer_preds[layer][1]>0 else 0
    #             if pred==hallu_cls:
    #                 layers_predicting.append(layer)
    #         layers_predicting= np.array(layers_predicting)
    #         if len(layers_predicting)>0:
    #             mc_val = np.min(probe_wise_entropy[layers_predicting]) # Most confident prediction
    #             mc_layers.append(np.min(np.argwhere(probe_wise_entropy==mc_val))) # np.min to find first most confident layer
    #         else:
    #             mc_layers.append(-1)
    #             no_hallu_preds += 1
    # print('Maxpool across tokens and using most confident probe:\n',classification_report(labels,confident_sample_pred))
    # precision, recall, thresholds = precision_recall_curve(labels, confident_sample_probs)
    # print('AUC:',auc(recall,precision))
    # print('\nMc Layer:\n',np.histogram(mc_layer, bins=range(num_layers+1)))
    # print('\nSamples with no hallucination prediction:',no_hallu_preds)
    # np.save(f'{args.save_path}/responses/best_layers/{args.model_name}_{args.dataset_name}_{args.responses_file_name}_mc_layers_tokenmax.npy', mc_layers)

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

        # Self-correct using token-wise aggregation and most confident
        final_labels = []
        for i,row in enumerate(labels):
            # Get prediction on orig response
            sample_preds = alltokens_preds[i]
            agg_layer_preds = []
            for layer_preds in sample_preds:
                agg_layer_preds.append(np.mean(layer_preds)) # Avg predictions across all tokens at a given layer
            agg_layer_preds = np.array(agg_layer_preds)
            agg_layer_preds = np.concatenate((1-agg_layer_preds[:, None], agg_layer_preds[:, None]),axis=1)
            probe_wise_entropy = (-agg_layer_preds*np.nan_to_num(np.log2(agg_layer_preds),neginf=0)).sum(axis=1)
            layer = np.argmin(probe_wise_entropy)
            orig_response_pred = 1 if agg_layer_preds[layer][1]>0 else 0 # Note this is already the distance from threshold, therefore we check for >0
            if orig_response_pred!=hallu_cls:
                final_labels.append(labels[i])
            else:
                final_labels.append(m_labels[i])
        print('\nDola after averaging across tokens and using most confident probe:',sum(final_labels)/len(final_labels) if hallu_cls==0 else 1-(sum(final_labels)/len(final_labels)))

        # Self-correct using token-wise aggregation and most confident
        final_labels = []
        for i,row in enumerate(labels):
            # Get prediction on orig response
            sample_preds = alltokens_preds[i]
            agg_layer_preds = []
            for layer_preds in sample_preds:
                agg_layer_preds.append(np.max(layer_preds)) # Maxpool predictions across all tokens at a given layer
            agg_layer_preds = np.array(agg_layer_preds)
            agg_layer_preds = np.concatenate((1-agg_layer_preds[:, None], agg_layer_preds[:, None]),axis=1)
            probe_wise_entropy = (-agg_layer_preds*np.nan_to_num(np.log2(agg_layer_preds),neginf=0)).sum(axis=1)
            layer = np.argmin(probe_wise_entropy)
            orig_response_pred = 1 if agg_layer_preds[layer][1]>0 else 0 # Note this is already the distance from threshold, therefore we check for >0
            if orig_response_pred!=hallu_cls:
                final_labels.append(labels[i])
            else:
                final_labels.append(m_labels[i])
        print('\nDola after maxpooling across tokens and using most confident probe:',sum(final_labels)/len(final_labels) if hallu_cls==0 else 1-(sum(final_labels)/len(final_labels)))

if __name__ == '__main__':
    main()