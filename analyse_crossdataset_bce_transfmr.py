import os
import re
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
def list_of_strs(arg):
    return list(map(str, arg.split(',')))

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
    parser.add_argument('--m_probes_file_name',default=None,type=list_of_strs,required=False,help='(default=%(default)s)')
    parser.add_argument("--probes_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--probes_file_name_concat", type=str, default='', help='local directory with dataset')
    parser.add_argument("--wp_probes_file_name", type=list_of_strs, default=None, help='local directory with dataset')
    parser.add_argument("--test_data", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--val_test_data", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--filt_testprompts_catg',type=list_of_ints, default=None)
    parser.add_argument('--test_num_samples',type=int, default=None)
    parser.add_argument('--wpdist_metric',type=str, default='')
    parser.add_argument('--lr_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument('--seed_list',default=None,type=list_of_ints,required=False,help='(default=%(default)s)')
    parser.add_argument('--sc_temp_list',default=[0],type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument("--skip_hypsearch", type=bool, default=False, help='')
    parser.add_argument('--layers_range_list',default=None,type=list_of_strs,required=False,help='(default=%(default)s)')
    parser.add_argument("--best_hyp_using_aufpr", type=bool, default=False, help='local directory with dataset')
    parser.add_argument("--best_hyp_using_trloss", type=bool, default=False, help='local directory with dataset')
    parser.add_argument("--best_threshold", type=bool, default=False, help='')
    parser.add_argument("--best_threshold_using_recall", type=bool, default=False, help='local directory with dataset')
    parser.add_argument('--fpr_at_recall',type=float, default=0.95)
    parser.add_argument('--aufpr_from',type=float, default=0.0)
    parser.add_argument('--aufpr_till',type=float, default=1.0)
    parser.add_argument("--min_max_scale_dist", type=bool, default=False, help='')
    parser.add_argument("--best_hyp_on_test", type=bool, default=False, help='')
    parser.add_argument("--show_val_res", type=bool, default=False, help='')
    parser.add_argument("--plot_loss", type=bool, default=False, help='')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()
    if args.model_name not in args.probes_file_name: raise ValueError("model name mismatch")

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
        if 'dola' in args.mitigated_responses_file_name:
            samples_neg_affected, samples_pos_affected = [], []
            with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
                data = json.load(read_file)
            with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file: # hl_llama_7B_{args.dataset_name} # {args.model_name}_{args.dataset_name}_baseline_responses_test.json
                ordered_data = json.load(read_file)
                for i in range(len(ordered_data['full_input_text'])):
                    # m_responses.append(data['model_answer'][i])
                    new_i = None
                    for j in range(len(data['full_input_text'])):
                        if data['full_input_text'][j]==ordered_data['full_input_text'][i]:
                            new_i = j
                            break
                    if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][new_i]==True else 0 # pos class is non-hallu
                    if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][new_i]==True else 1 # pos class is hallu
                    m_labels.append(label)
            #         if labels[i]!=hallu_cls and label==hallu_cls: samples_neg_affected.append(i)
            #         if labels[i]==hallu_cls and label!=hallu_cls: samples_pos_affected.append(i)
            # print('Num of samples negatively affected:',len(samples_neg_affected))
            # print('Num of samples positively affected:',len(samples_pos_affected))
        elif 'sampled' in args.mitigated_responses_file_name:
            if 'strqa' in args.dataset_name:
                with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
                    data = json.load(read_file)
                for i in range(len(data['full_input_text'])):
                    for j in [1]: # we only want the first random sample
                        if 'hallu_pos' not in args.probes_file_name: label = 1 if data['is_correct'][i][j]==True else 0
                        if 'hallu_pos' in args.probes_file_name: label = 0 if data['is_correct'][i][j]==True else 1
                        m_labels.append(label)
            elif 'trivia' in args.dataset_name:
                with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.mitigated_responses_file_name}.json', 'r') as read_file:
                    for line in read_file:
                        data = json.loads(line)
                        for j in [1]: # we only want the first random sample
                            if 'hallu_pos' not in args.probes_file_name: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.probes_file_name: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            m_labels.append(label)
    
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

    all_results_list = []

    for seed_i,seed in enumerate(args.seed_list):
        args.probes_file_name = 'T'+str(seed)+'_'+args.probes_file_name.split('_',1)[1]
        if args.layers_range_list is not None:
            args.probes_file_name = re.sub("hallu_pos_[0-9]+_[0-9]+_[0-9]+","hallu_pos_"+args.layers_range_list[seed_i],args.probes_file_name)
        seed_results_list = []
        if args.m_probes_file_name is not None: seed_m_probes_file_name = args.m_probes_file_name[seed_i]
        if args.wp_probes_file_name is not None: seed_wp_probes_file_name = args.wp_probes_file_name[seed_i]

        # val_pred_model,all_val_true[fold][0]
        def my_aufpr(preds,labels,getfull=False):
            preds, labels = np.squeeze(preds), np.squeeze(labels)
            r_list, fpr_list = [], []
            # print(np.histogram(preds, bins='sqrt'))
            # preds = (preds - preds.min()) / (preds.max() - preds.min()) # min-max-scale distances # not required as we already do this before calling the func
            # print(np.histogram(preds))
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
            fpr_at_recall_vals = np.interp(check_recall_intervals, recall_vals, fpr_at_recall_vals)
            if getfull:
                aufpr = auc(check_recall_intervals,fpr_at_recall_vals)
            else:
                check_recall_intervals,fpr_at_recall_vals = np.array(check_recall_intervals), np.array(fpr_at_recall_vals)
                aufpr_idxes = (check_recall_intervals>=args.aufpr_from) & (check_recall_intervals<=args.aufpr_till)
                aufpr = auc(check_recall_intervals[aufpr_idxes],fpr_at_recall_vals[aufpr_idxes])
            return check_recall_intervals, fpr_at_recall_vals, aufpr

        def results_at_best_lr(model,seed_i):
            if args.skip_hypsearch:
                lr_search_list = [args.lr_list[seed_i]] # One-to-one mapping of seed and lr
                supcon_temp_search_list = [args.sc_temp_list[seed_i]] if len(args.sc_temp_list)==len(args.lr_list) else args.sc_temp_list # One-to-one mapping of seed and supcon_temp
            else:
                lr_search_list,supcon_temp_search_list = args.lr_list,args.sc_temp_list
            if lr_search_list is not None:
                probes_file_name_list, perf_by_lr = [], []
                for lr in args.lr_list:
                    for temp in supcon_temp_search_list:
                        if temp==0:
                            probes_file_name = args.probes_file_name
                        else:
                            if temp==0.1: 
                                temp=''
                            else:
                                temp = str(temp) + '_'
                            # temp = str(temp) + '_'
                            search_str = 'hallu_pos_valaug_' if 'valaug' in args.probes_file_name else 'hallu_pos_'
                            fn_left_text = args.probes_file_name.split(search_str,1)[0] + search_str
                            fn_right_text = args.probes_file_name.split(search_str,1)[1].split('_',1)[1]
                            probes_file_name = fn_left_text + temp + fn_right_text
                            # print(probes_file_name)
                        probes_file_name = probes_file_name + str(lr) + '_False' + args.probes_file_name_concat
                        val_probes_file_name = probes_file_name if args.val_test_data is None else probes_file_name.replace(args.test_data,args.val_test_data)
                        try:
                            if args.best_hyp_on_test:
                                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{probes_file_name}_test_true.npy')
                            else:
                                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{val_probes_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{val_probes_file_name}_val_true.npy')
                        except FileNotFoundError:
                            probes_file_name = probes_file_name.replace("/","")
                            val_probes_file_name = val_probes_file_name.replace("/","")
                            if args.best_hyp_on_test:
                                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{probes_file_name}_test_true.npy')
                            else:
                                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{val_probes_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{val_probes_file_name}_val_true.npy')
                        probes_file_name_list.append(probes_file_name)
                        # print(all_val_pred.shape)
                        if args.min_max_scale_dist: all_val_pred[0][model] = (all_val_pred[0][model] - all_val_pred[0][model].min()) / (all_val_pred[0][model].max() - all_val_pred[0][model].min()) # min-max-scale distances
                        auc_val = roc_auc_score(all_val_true[0][model], [-v for v in np.squeeze(all_val_pred[0][model])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(all_val_true[0][model], np.squeeze(all_val_pred[0][model]))
                        _, _, aufpr_val = my_aufpr(all_val_pred[0][model],all_val_true[0][model],getfull=True)
                        trloss_probes_file_name = probes_file_name.replace('kmeans_','').replace('mahalanobis_centers1_','').replace('_bestusinglast','')
                        train_loss = np.load(f'{args.save_path}/probes/{trloss_probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()[0][model][-1] if args.best_hyp_using_trloss else 0  # index fold, model, epoch  
                        perf = aufpr_val if args.best_hyp_using_aufpr else train_loss if args.best_hyp_using_trloss else auc_val 
                        perf_by_lr.append(perf)
                best_probes_file_name = probes_file_name_list[np.argmin(perf_by_lr)] if (args.best_hyp_using_aufpr or args.best_hyp_using_trloss) else probes_file_name_list[np.argmax(perf_by_lr)]
                print(best_probes_file_name)
            else:
                best_probes_file_name = args.probes_file_name
            
            if args.plot_loss:
                # Create dirs if does not exist:
                # if not os.path.exists(f'{args.save_path}/loss_figures/{best_probes_file_name}'):
                #     os.makedirs(f'{args.save_path}/loss_figures/{best_probes_file_name}', exist_ok=True)
                if not os.path.exists(f'/home/jovyan/loss_figures/{best_probes_file_name}'):
                    os.makedirs(f'/home/jovyan/loss_figures/{best_probes_file_name}', exist_ok=True)
                # loss_to_plot = np.load(f'{args.save_path}/probes/{best_probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()
                # loss_to_plot1 = np.load(f'{args.save_path}/probes/{best_probes_file_name}_supcon1_train_loss.npy', allow_pickle=True).item()
                # loss_to_plot2 = np.load(f'{args.save_path}/probes/{best_probes_file_name}_supcon2_train_loss.npy', allow_pickle=True).item()
                # # print(loss_to_plot[0])
                # fig, axs = plt.subplots(1,1)
                # axs.plot(loss_to_plot[0][0],label='total') # index fold, model
                # axs.plot(loss_to_plot1[0][0],label='pos')
                # axs.plot(loss_to_plot2[0][0],label='wp')
                # axs.legend()
                # fig.savefig(f'{args.save_path}/loss_figures/{best_probes_file_name}_supcon_train_loss.png')
                loss_to_plot = np.load(f'{args.save_path}/probes/{best_probes_file_name}_train_loss.npy', allow_pickle=True).item()
                loss_to_plot1 = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_loss.npy', allow_pickle=True).item()
                loss_to_plot2 = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_auc.npy', allow_pickle=True).item()
                fig, axs = plt.subplots(1,1)
                axs.plot(loss_to_plot[0][0],label='train_ce_loss') # index fold, model
                axs.plot(loss_to_plot1[0][0],label='val_ce_loss')
                axs.plot(loss_to_plot2[0][0],label='val_auc')
                axs.legend()
                # fig.savefig(f'{args.save_path}/loss_figures/{best_probes_file_name}_train_curves.png')
                fig.savefig(f'/home/jovyan/loss_figures/{best_probes_file_name}_train_curves.png')

            if args.best_hyp_on_test:
                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_pred.npy'), np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_true.npy')
            else:
                best_val_probes_file_name = best_probes_file_name if args.val_test_data is None else best_probes_file_name.replace(args.test_data,args.val_test_data)
                all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{best_val_probes_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{best_val_probes_file_name}_val_true.npy')
            if args.best_threshold:
                best_val_perf, best_t = 0, 0.5
                # best_val_fpr, best_t = 1, 0
                # thresholds = np.histogram_bin_edges(all_val_pred[fold][model], bins='sqrt') if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else [x / 100.0 for x in range(0, 105, 5)]
                # print(np.histogram(all_val_pred[fold][model], bins='sqrt'))
                val_dist_min, val_dist_max = all_val_pred[fold][model].min(), all_val_pred[fold][model].max()
                if args.min_max_scale_dist: all_val_pred[fold][model] = (all_val_pred[fold][model] - val_dist_min) / (val_dist_max - val_dist_min) # min-max-scale distances
                if (('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name)) and args.min_max_scale_dist==False:
                    thresholds = np.histogram_bin_edges(all_val_pred[fold][model], bins='sqrt')
                else:
                    thresholds = [x / 100.0 for x in range(0, 105, 5)]
                for t in thresholds:
                    val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
                    if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                        val_pred_model[all_val_pred[fold][model]<=t] = 1 # <= to ensure correct classification when dist = [-1,0]
                        val_pred_model[all_val_pred[fold][model]>t] = 0
                    else:
                        val_pred_model[all_val_pred[fold][model]>t] = 1
                        val_pred_model[all_val_pred[fold][model]<=t] = 0
                    # print(all_val_true[fold][0].shape,val_pred_model.shape,f1_score(all_val_true[fold][0],val_pred_model),f1_score(all_val_true[fold][0],np.squeeze(val_pred_model)))
                    # sys.exit()
                    cls1_f1 = f1_score(all_val_true[fold][0],val_pred_model)
                    cls0_f1 = f1_score(all_val_true[fold][0],val_pred_model,pos_label=0)
                    recall = recall_score(all_val_true[fold][0],val_pred_model)
                    perf = recall if args.best_threshold_using_recall else np.mean((cls1_f1,cls0_f1)) # cls1_f1
                    if perf>best_val_perf:
                        best_val_perf, best_t = perf, t
                    # fp = np.sum((np.squeeze(val_pred_model) == 1) & (np.squeeze(all_val_true[fold][0]) == 0))
                    # tn = np.sum((np.squeeze(val_pred_model) == 0) & (np.squeeze(all_val_true[fold][0]) == 0))
                    # val_fpr = fp / (fp + tn)
                    # if recall >= 0.9:
                    #     if val_fpr<best_val_fpr:
                    #         best_val_fpr, best_t = val_fpr, t
                    # print(recall)
            else:
                best_t = 0.5
            print('best-t:', best_t)
            return best_probes_file_name, all_val_pred, all_val_true, best_t, val_dist_min, val_dist_max

        # all_val_pred, all_val_true = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_pred.npy'), np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_true.npy')
        fold = 0
        test_f1_cls0, test_f1_cls1, best_r, test_fpr_best_r, test_fpr_best_f1, test_fpr, test_recall_cls0, test_recall_cls1, test_precision_cls1, val_f1_cls1, val_f1_cls0, val_f1_avg = [], [], [], [], [], [], [], [], [], [], [], []
        best_probes_per_model, layer_pred_thresholds = [], []
        excl_layers, incl_layers = [], []
        aupr_by_layer, auroc_by_layer = [], []
        num_models = 1 # 33 if args.using_act=='layer' else 32 if args.using_act=='mlp' else 32*32
        print(num_models)
        all_preds = []
        for model in tqdm(range(num_models)):
            best_probes_file_name, all_val_pred, all_val_true, best_t, val_dist_min, val_dist_max  = results_at_best_lr(model,seed_i)
            best_probes_per_model.append(best_probes_file_name)
            layer_pred_thresholds.append(best_t)
            if args.show_val_res:
                test_preds = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_pred.npy')[0]
                labels = np.load(f'{args.save_path}/probes/{best_probes_file_name}_val_true.npy')[0][0] ## Since labels are same for all models
            else:
                test_preds = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_pred.npy')[0]
                labels = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_true.npy')[0][0] ## Since labels are same for all models
                if args.wpdist_metric!='': 
                    wp_dist = np.load(f'{args.save_path}/probes/{best_probes_file_name}_test_wpdist_{args.wpdist_metric}.npy')[0][0]
                    # wp_dist = wp_dist/2
                    # print(wp_dist.shape, wp_dist[:10,:], np.min(wp_dist[:10,:],axis=1),np.max( wp_dist[:10,:],axis=1))
                if args.m_probes_file_name is not None: 
                    m_test_pred_model = np.load(f'{args.save_path}/probes/{seed_m_probes_file_name}_test_pred_model.npy')
                    if 'sampled' in args.mitigated_responses_file_name:
                        if args.use_all_m_sampled_responses:
                            m_prompts = len(m_test_pred_model)/args.test_num_samples
                            m_test_pred_model_temp= []
                            print(m_test_pred_model[:20])
                            for m_prompt_idx in range(m_prompts[:2]):
                                m_idx = m_prompt_idx*args.test_num_samples
                                sample_probs = m_test_pred_model[m_idx:m_idx+args.test_num_samples]
                                print(sample_probs)
                                m_test_pred_model_temp.append(np.min(sample_probs))
                            sys.exit()
                            m_test_pred_model = m_test_pred_model_temp
                        else:
                            first_random_idx = np.arange(0,len(m_test_pred_model),args.test_num_samples)
                            m_test_pred_model = m_test_pred_model[first_random_idx]
                        try:
                            assert len(m_test_pred_model)==len(labels)
                        except AssertionError:
                            print(len(m_test_pred_model),len(labels))
                            sys.exit()
            
            if args.filt_testprompts_catg is not None:
                num_prompts = int(len(test_preds[0])/args.test_num_samples)
                print('\n\ntest_preds shape:',test_preds.shape,' num_prompts:',num_prompts)
                select_instances, num_prompts_in_catg = [], 0
                for k in range(num_prompts):
                    cur_prompt_idx = k*args.test_num_samples
                    sample_dist = sum(labels[cur_prompt_idx:cur_prompt_idx+args.test_num_samples])
                    # print(labels[cur_prompt_idx:cur_prompt_idx+args.test_num_samples])
                    # if k==2: sys.exit()
                    if sample_dist==args.test_num_samples and 0 in args.filt_testprompts_catg: #0
                        select_instances += list(np.arange(cur_prompt_idx,cur_prompt_idx+args.test_num_samples,1))
                        num_prompts_in_catg += 1
                    elif sample_dist==0  and 1 in args.filt_testprompts_catg: #1
                        select_instances += list(np.arange(cur_prompt_idx,cur_prompt_idx+args.test_num_samples,1))
                        num_prompts_in_catg += 1
                    elif sample_dist>0 and sample_dist <= int(args.test_num_samples/3) and 2 in args.filt_testprompts_catg: #2
                        select_instances += list(np.arange(cur_prompt_idx,cur_prompt_idx+args.test_num_samples,1))
                        num_prompts_in_catg += 1
                    elif sample_dist > int(2*args.test_num_samples/3) and sample_dist<args.test_num_samples and 3 in args.filt_testprompts_catg: #3
                        select_instances += list(np.arange(cur_prompt_idx,cur_prompt_idx+args.test_num_samples,1))
                        num_prompts_in_catg += 1
                    elif sample_dist > int(args.test_num_samples/3) and sample_dist <= int(2*args.test_num_samples/3) and 4 in args.filt_testprompts_catg: #4
                        select_instances += list(np.arange(cur_prompt_idx,cur_prompt_idx+args.test_num_samples,1))
                        num_prompts_in_catg += 1
                select_instances = np.array(select_instances)
                test_preds, labels = test_preds[:,select_instances], labels[select_instances]
                if args.wpdist_metric!='': wp_dist = wp_dist[select_instances]
                print('test_preds shape:',test_preds.shape,' labels shape:',labels.shape)
                print('num_prompts_in_catg:',num_prompts_in_catg,'\n\n')
                # print('wp_dist:',np.min(wp_dist[:,1]),np.max( wp_dist[:,1]))
            
            val_pred_model = deepcopy(all_val_pred[fold][model]) # Deep copy so as to not touch orig values
            if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                val_pred_model[all_val_pred[fold][model]<=best_t] = 1 # <= to ensure correct classification when dist = [-1,0]
                val_pred_model[all_val_pred[fold][model]>best_t] = 0
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
            print(np.histogram(test_preds[model]))
            if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name):
                test_pred_model[test_preds[model]<=best_t] = 1 # <= to ensure correct classification when dist = [-1,0]
                test_pred_model[test_preds[model]>best_t] = 0
            else:
                test_pred_model[test_preds[model]>best_t] = 1
                test_pred_model[test_preds[model]<=best_t] = 0
                np.save(f'{args.save_path}/probes/{best_probes_file_name}_test_pred_model.npy',test_pred_model)
            # print(recall_score(labels,test_pred_model),recall_score(labels,np.squeeze(test_pred_model)))
            cls1_f1, cls1_re, cls1_pr = f1_score(labels,test_pred_model), recall_score(labels,test_pred_model), precision_score(labels,test_pred_model)
            cls0_f1, cls0_re = f1_score(labels,test_pred_model,pos_label=0), recall_score(labels,test_pred_model,pos_label=0)
            test_f1_cls0.append(cls0_f1)
            test_f1_cls1.append(cls1_f1)
            test_recall_cls0.append(cls0_re)
            test_recall_cls1.append(cls1_re)
            test_precision_cls1.append(cls1_pr)
            precision, recall, _ = precision_recall_curve(labels, [-v for v in np.squeeze(test_preds[model])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else precision_recall_curve(labels, np.squeeze(test_preds[model]))
            aupr_by_layer.append(auc(recall,precision))
            if args.filt_testprompts_catg!=[0] and args.filt_testprompts_catg!=[1]: 
                auc_val = roc_auc_score(labels, [-v for v in np.squeeze(test_preds[model])]) if ('knn' in args.probes_file_name) or ('kmeans' in args.probes_file_name) else roc_auc_score(labels, np.squeeze(test_preds[model]))
            else:
                auc_val = 0
            auroc_by_layer.append(auc_val)

            test_pred_model = np.squeeze(test_pred_model)
            fp = np.sum((test_pred_model == 1) & (labels == 0))
            tn = np.sum((test_pred_model == 0) & (labels == 0))
            test_fpr_best_f1.append(fp / (fp + tn))

            # check_indexes = np.array([   3,  11,  12,  17,  24, 34,  36,  45,  46,  49,  51,  52,  54,  56,  58,  64,  65,  67])
            # fixes = (test_pred_model[check_indexes] == 1)
            # fixes_index = np.where(fixes)[0]
            # print('% fixes:',len(fixes_index)/len(check_indexes))
            # print('Index of fixes:',check_indexes[fixes_index])
            # sys.exit()

        # print('\nValidation performance:\n',val_f1_avg)
        incl_layers = np.array(incl_layers)
        print('\nExcluded layers:',excl_layers)
        # print(incl_layers)
        # if 'hallu_pos' in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls0),np.mean(test_f1_cls1),'\n') # NH, H
        # if 'hallu_pos' not in args.probes_file_name: print('\nAverage F1:',np.mean(test_f1_cls1),np.mean(test_f1_cls0),'\n') # NH, H
        # if 'hallu_pos' in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls0),np.mean(test_recall_cls1),'\n') # NH, H
        # if 'hallu_pos' not in args.probes_file_name: print('\nAverage Recall:',np.mean(test_recall_cls1),np.mean(test_recall_cls0),'\n') # NH, H
        
        seed_results_list.append(np.mean([np.mean(test_f1_cls0),np.mean(test_f1_cls1)])*100) # print(np.mean([np.mean(test_f1_cls0),np.mean(test_f1_cls1)]))
        # seed_results_list.append(np.mean(best_r))
        # seed_results_list.append(np.mean(test_fpr_best_r))
        if args.fpr_at_recall==-1:
            # Create dirs if does not exist:
            if not os.path.exists(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}'):
                os.makedirs(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}', exist_ok=True)
            # print('model:',model)
            recall_vals, fpr_at_recall_vals, aucfpr = my_aufpr(test_preds[model],labels)
            fig, axs = plt.subplots(1,1)
            axs.plot(recall_vals,fpr_at_recall_vals)
            for xy in zip(recall_vals,fpr_at_recall_vals):
                axs.annotate('(%.2f, %.2f)' % xy, xy=xy)
            axs.set_xlabel('Recall')
            axs.set_ylabel('FPR')
            axs.title.set_text('FPR at recall')
            fig.savefig(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall.png')
            seed_results_list.append(aucfpr*100)
            np.save(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall_xaxis.npy',np.array(recall_vals))
            np.save(f'{args.save_path}/fpr_at_recall_curves/{best_probes_file_name}_fpr_at_recall_yaxis.npy',np.array(fpr_at_recall_vals))
        else:
            seed_results_list.append(np.mean(test_fpr)*100)
        seed_results_list.append(np.mean(test_fpr_best_f1)*100)
        seed_results_list.append(np.mean(test_f1_cls1)*100)
        seed_results_list.append(np.mean(test_precision_cls1)*100) # print(np.mean(test_precision_cls1)) # H
        seed_results_list.append(np.mean(test_recall_cls1)*100) # print(np.mean(test_recall_cls1)) # H
        seed_results_list.append(np.mean(aupr_by_layer)*100) # print(np.mean(aupr_by_layer)) # 'Avg AUPR:',
        seed_results_list.append(np.mean(auroc_by_layer)*100) # print(np.mean(auroc_by_layer)) # 'Avg AUROC:',
        if args.wpdist_metric!='':
            use_indices = wp_dist[:,0]!=-10000 # Only use samples which have at least one other wp sample of same class
            wp_dist[:,0][wp_dist[:,0]<0]=0 # Fix cases where cosine_sim results in values>1 # This is an open issue with torch
            seed_results_list.append(np.mean(wp_dist[use_indices,0])) # Dist to same class

            seed_results_list.append(np.mean(wp_dist[:,1])) # Dist to opp class
            
            r_dist = wp_dist[use_indices,1]/(wp_dist[use_indices,0] + wp_dist[use_indices,1])
            seed_results_list.append(np.mean(r_dist)) # Dist to opp class, relative to same class (within prompt)   
        if args.wp_probes_file_name is not None:
            wp_dist = np.load(f'{args.save_path}/probes/{seed_wp_probes_file_name}_test_wpdist_cosine_individual.npy')[0][0]
            wp_dist[wp_dist<0]=0 # Fix cases where cosine_sim results in values>1 # This is an open issue with torch
            k = 10
            lamb1 = 0.5
            lamb2 = 1-lamb1
            sample_idxs = []
            first_sample_idxs = np.arange(0,len(wp_dist),args.test_num_samples)
            wp_dist = wp_dist/2 # (cosine distance)/2 gives probability
            wp_dist_k = []
            for sample_idx in first_sample_idxs:
                sample_idxs_k = []
                for next_idx in range(k):
                    sample_idxs_k.append(sample_idx + next_idx)
                wp_dist_k.append(np.max(wp_dist[np.array(sample_idxs_k),:k-1])) # max dist among pairwise distances of k samples
            seed_results_list.append(roc_auc_score(labels, wp_dist_k))
            ensemble_prob = (lamb1*np.squeeze(test_preds[model]) + lamb2*np.array(wp_dist_k)) # weighted avg
            seed_results_list.append(roc_auc_score(labels, ensemble_prob))
        
        # print(auroc_by_layer)
        if args.mitigated_responses_file_name!='':
            print('\n\nOriginal perf:',sum(labels)/len(labels) if hallu_cls==0 else 1-(sum(labels)/len(labels)))
            print('\n\nDoLa perf:',sum(m_labels)/len(m_labels) if hallu_cls==0 else 1-(sum(m_labels)/len(m_labels)))
            samples_neg_affected, samples_pos_affected = 0, 0
            for i,row in enumerate(labels):
                if labels[i]!=hallu_cls and m_labels[i]==hallu_cls: samples_neg_affected += 1
                if labels[i]==hallu_cls and m_labels[i]!=hallu_cls: samples_pos_affected += 1
            print('Num of samples positively affected:',samples_pos_affected*100/len(labels))
            print('Num of samples negatively affected:',samples_neg_affected*100/len(labels))

            # Self-correct using CLAP pred
            final_labels1, final_labels1b, nh_among_abs1b, labels2, final_labels2, nh_among_abs = [], [], 0, [], [], 0
            for i,row in enumerate(labels):
                # Get prediction on orig response
                orig_response_pred = test_pred_model[i] # Get predictions of all samples (we have only one model when using CLAP)
                if orig_response_pred!=hallu_cls:
                    final_labels1.append(labels[i])
                    final_labels1b.append(labels[i])
                else:
                    if labels[i]!=hallu_cls: nh_among_abs1b += 1
                    final_labels1.append(m_labels[i])
                if args.m_probes_file_name is not None:
                    m_response_pred = m_test_pred_model[i]
                    if orig_response_pred!=hallu_cls:
                        final_labels2.append(labels[i])
                        labels2.append(labels[i])
                    elif m_response_pred!=hallu_cls:
                        final_labels2.append(m_labels[i])
                        labels2.append(labels[i])
                    else:
                        if labels[i]!=hallu_cls: nh_among_abs += 1
                        pass # In this case, we abstain (either prediction is hallucination)
            new_perf1b = sum(final_labels1b)/len(final_labels1b) if hallu_cls==0 else 1-(sum(final_labels1b)/len(final_labels1b))
            # print('\nDola after using last layer:',new_perf)
            seed_results_list.append(new_perf1b*100)
            num_abs = len(labels)-len(final_labels1b)
            seed_results_list.append(num_abs*100/len(labels))
            seed_results_list.append(nh_among_abs1b*100/len(labels))
            # print('Num of samples abstained:',num_abs*100/len(labels))
            new_perf1 = sum(final_labels1)/len(final_labels1) if hallu_cls==0 else 1-(sum(final_labels1)/len(final_labels1))
            # print('\nDola after using last layer:',new_perf)
            seed_results_list.append(new_perf1*100)
            samples_neg_affected, samples_pos_affected = 0, 0
            for i,row in enumerate(labels):
                if labels[i]!=hallu_cls and final_labels1[i]==hallu_cls: samples_neg_affected += 1
                if labels[i]==hallu_cls and final_labels1[i]!=hallu_cls: samples_pos_affected += 1
            seed_results_list.append(samples_pos_affected*100/len(labels))
            seed_results_list.append(samples_neg_affected*100/len(labels))
            # print('Num of samples positively affected:',samples_pos_affected*100/len(labels))
            # print('Num of samples negatively affected:',samples_neg_affected*100/len(labels))
            if args.m_probes_file_name is not None:
                new_perf2 = sum(final_labels2)/len(final_labels2) if hallu_cls==0 else 1-(sum(final_labels2)/len(final_labels2))
                seed_results_list.append(new_perf2*100)
                samples_neg_affected, samples_pos_affected = 0, 0
                for i,row in enumerate(labels2):
                    if labels2[i]!=hallu_cls and final_labels2[i]==hallu_cls: samples_neg_affected += 1
                    if labels2[i]==hallu_cls and final_labels2[i]!=hallu_cls: samples_pos_affected += 1
                seed_results_list.append(samples_pos_affected*100/len(labels))
                seed_results_list.append(samples_neg_affected*100/len(labels))
                # print('Num of samples positively affected:',samples_pos_affected*100/len(labels))
                # print('Num of samples negatively affected:',samples_neg_affected*100/len(labels))
                num_abs = len(labels)-len(final_labels2)
                seed_results_list.append(num_abs*100/len(labels))
                seed_results_list.append(nh_among_abs*100/len(labels))
                # print('Num of samples abstained:',num_abs*100/len(labels))
                # print('%NH among abstained:',nh_among_abs*100/len(labels))
        
        all_preds = np.stack(all_preds, axis=0)

        all_results_list.append(np.array(seed_results_list))
    print(all_results_list)
    print(', '.join(map(str,np.mean(np.stack(all_results_list),axis=0).tolist())))
    print(', '.join(map(str,np.std(np.stack(all_results_list),axis=0).tolist())))

if __name__ == '__main__':
    main()