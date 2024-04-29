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
from utils import LogisticRegression_Torch

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def get_probe_wgts(fold,model,results_file_name,save_path):
    act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128}
    using_act = 'ah' if '_ah_' in results_file_name else 'mlp_l1' if '_mlp_l1_' in results_file_name else 'mlp'
    num_layers = 32 if '_7B_' in results_file_name else 40 if '_13B_' in results_file_name else 60
    layer = model if using_act=='mlp' else np.floor(model/num_layers) # 0 to 31 -> 0, 32 to 63 -> 1, etc.
    head = 0 if using_act=='mlp' else (model%num_layers) 
    current_linear_model = LogisticRegression_Torch(act_dims[using_act], 2)
    linear_model = torch.load(f'{save_path}/probes/models/{results_file_name}_model{fold}_{layer}_{head}')
    return linear_model.linear.weight[0], linear_model.linear.weight[1]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--layer_starts',default=None,type=list_of_ints,help='(default=%(default)s)')
    parser.add_argument('--layer_ends',default=None,type=list_of_ints,help='(default=%(default)s)')
    parser.add_argument("--responses_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--use_similarity", type=bool, default=False)
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

    if args.layer_ends is None:
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
        all_val_logits, all_test_logits = np.load(f'{args.save_path}/probes/{args.results_file_name}_val_logits.npy'), np.load(f'{args.save_path}/probes/{args.results_file_name}_test_logits.npy')
        if args.use_similarity:
            sim_file_name = args.results_file_name.replace('individual_linear','individual_linear_unitnorm') if 'individual_linear_unitnorm' not in args.results_file_name else args.results_file_name
            all_val_sim, all_test_sim = np.load(f'{args.save_path}/probes/{sim_file_name}_val_sim.npy'), np.load(f'{args.save_path}/probes/{sim_file_name}_test_sim.npy')
    else:
        all_val_loss_files =[np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_val_loss.npy',allow_pickle=True).item() for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)]
        # all_train_loss = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_train_loss.npy',allow_pickle=True).item() for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_test_f1s = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_test_f1.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_val_f1s = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_val_f1.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_test_pred = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_test_pred.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_test_true = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_test_true.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_val_pred = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_val_pred.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_val_true = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_val_true.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_val_logits = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_val_logits.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        all_test_logits = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{layer_start}_{layer_end}_test_logits.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        # print(all_test_logits.shape)
        # exit()

    np.set_printoptions(precision=1)
    np.seterr(divide = 'ignore') # turn off for display clarity

    for fold in range(len(all_test_f1s)):

        for model in range(all_test_true[fold].shape[0]):
            assert sum(all_test_true[fold][0]==all_test_true[fold][model])==len(all_test_true[fold][0]) # check all models have same batch order
            assert sum(all_val_true[fold][0]==all_val_true[fold][model])==len(all_val_true[fold][0])

        test_f1_using_logits, test_f1_cls0_using_logits, test_f1_avg_using_logits, val_f1_using_logits, val_f1_cls0_using_logits, val_f1_avg_using_logits = [], [], [], [], [], []
        test_f1_cls0, test_f1_avg, val_f1_cls0, val_f1_avg = [], [], [], []
        for model in range(all_test_logits[fold].shape[0]):
            cls1_f1 = f1_score(all_test_true[fold][0],np.argmax(all_test_logits[fold][model], axis=1))
            cls0_f1 = f1_score(all_test_true[fold][0],np.argmax(all_test_logits[fold][model], axis=1),pos_label=0)
            test_f1_using_logits.append(cls1_f1)
            test_f1_cls0_using_logits.append(cls0_f1)
            test_f1_avg_using_logits.append(np.mean((cls1_f1,cls0_f1)))
            #
            cls1_f1 = f1_score(all_val_true[fold][0],np.argmax(all_val_logits[fold][model], axis=1))
            cls0_f1 = f1_score(all_val_true[fold][0],np.argmax(all_val_logits[fold][model], axis=1),pos_label=0)
            val_f1_using_logits.append(cls1_f1)
            val_f1_cls0_using_logits.append(cls0_f1)
            val_f1_avg_using_logits.append(np.mean((cls1_f1,cls0_f1)))
            #
            cls1_f1 = f1_score(all_test_true[fold][0],np.argmax(all_test_pred[fold][model], axis=1))
            cls0_f1 = f1_score(all_test_true[fold][0],np.argmax(all_test_pred[fold][model], axis=1),pos_label=0)
            test_f1_cls0.append(cls0_f1)
            test_f1_avg.append(np.mean((cls1_f1,cls0_f1)))
            #
            cls1_f1 = f1_score(all_val_true[fold][0],np.argmax(all_val_pred[fold][model], axis=1))
            cls0_f1 = f1_score(all_val_true[fold][0],np.argmax(all_val_pred[fold][model], axis=1),pos_label=0)
            val_f1_cls0.append(cls0_f1)
            val_f1_avg.append(np.mean((cls1_f1,cls0_f1)))
            

        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold]),np.mean(test_f1_cls0[fold]))
        print('Best:',all_test_f1s[fold][np.argmax(val_f1_avg)],test_f1_cls0[np.argmax(val_f1_avg)])
        print('Average using logits:',np.mean(test_f1_using_logits),np.mean(test_f1_cls0_using_logits))
        print('Best using logits:',test_f1_using_logits[np.argmax(val_f1_avg_using_logits)],test_f1_cls0_using_logits[np.argmax(val_f1_avg_using_logits)])
        
        # Oracle 1
        print('\n')
        best_sample_pred =[]
        num_correct_probes_nonhallu, correct_probes_nonhallu, correct_probes_nonhallu_sets1, correct_probes_nonhallu_sets2 = [], [], [], []
        num_correct_probes_hallu, correct_probes_hallu, correct_probes_hallu_sets1, correct_probes_hallu_sets2 = [], [], [], []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.argmax(sample_pred,axis=1)
            # assert sample_pred.shape==(32,) # num_layers
            correct_answer = all_test_true[fold][0][i]
            if correct_answer==1: num_correct_probes_nonhallu.append(sum(sample_pred==correct_answer))
            if correct_answer==1 and sum(sample_pred==correct_answer)<20: correct_probes_nonhallu += [idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]
            if correct_answer==1 and sum(sample_pred==correct_answer)>10: correct_probes_nonhallu_sets1.append(set([idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]))
            if correct_answer==1 and sum(sample_pred==correct_answer)>5: correct_probes_nonhallu_sets2.append(set([idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]))
            if correct_answer==0: num_correct_probes_hallu.append(sum(sample_pred==correct_answer))
            if correct_answer==0 and sum(sample_pred==correct_answer)<20: correct_probes_hallu += [idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]
            if correct_answer==0 and sum(sample_pred==correct_answer)>10: correct_probes_hallu_sets1.append(set([idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]))
            if correct_answer==0 and sum(sample_pred==correct_answer)>5: correct_probes_hallu_sets2.append(set([idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]))
            # if i==0: print(sample_pred==correct_answer,sum(sample_pred==correct_answer))
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        assert f1_score(all_test_true[fold][0],all_test_true[fold][0])==1
        fig, axs = plt.subplots(2,2)
        counts, bins = np.histogram(num_correct_probes_nonhallu, bins=range(33))
        axs[0,0].stairs(counts, bins)
        axs[0,0].title.set_text('Non-Hallucinated')
        axs[0,0].set_xlabel('# probes classifying correctly')
        axs[0,0].set_ylabel('# samples')
        counts, bins = np.histogram(num_correct_probes_hallu, bins=range(33))
        axs[0,1].stairs(counts, bins)
        axs[0,1].title.set_text('Hallucinated')
        axs[0,1].set_xlabel('# probes classifying correctly')
        counts, bins = np.histogram(correct_probes_nonhallu, bins=range(33))
        axs[1,0].stairs(counts, bins)
        axs[1,0].set_xlabel('probe idx')
        axs[1,0].set_ylabel('# samples correct\n(when num_correct_probes<20)')
        counts, bins = np.histogram(correct_probes_hallu, bins=range(33))
        axs[1,1].stairs(counts, bins)
        axs[1,1].set_xlabel('probe idx')
        fig.tight_layout()
        fig.savefig(f'{args.save_path}/figures/{args.results_file_name}_oracle_hist.png')
        print('Oracle:',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
        num_correct_probes_nonhallu = np.array(num_correct_probes_nonhallu)
        num_correct_probes_hallu = np.array(num_correct_probes_hallu)
        print('Non-Hallucinated hard samples:',sum(num_correct_probes_nonhallu<20),sum(num_correct_probes_nonhallu<10),sum(num_correct_probes_nonhallu<5))
        print('Hallucinated hard samples:',sum(num_correct_probes_hallu<20),sum(num_correct_probes_hallu<10),sum(num_correct_probes_hallu<5))
        print('# Hallucinated samples:',sum(all_test_true[fold][0]==0))
        print('Set intersection for >10 probes:',set.intersection(*correct_probes_nonhallu_sets1),set.intersection(*correct_probes_hallu_sets1))
        print('Set intersection for >5 probes:',set.intersection(*correct_probes_nonhallu_sets2),set.intersection(*correct_probes_hallu_sets2))
        print(set.intersection(*correct_probes_nonhallu_sets1[:5]))
        
        # Oracle 2
        best_sample_pred =[]
        best_probes_nonhallu, correct_probes_nonhallu = [], []
        best_probes_hallu, correct_probes_hallu = [], []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            best_probe_idxs = np.argpartition(probe_wise_entropy, -5)[-5:]
            top_5_lower_bound_val = np.min(probe_wise_entropy[best_probe_idxs])
            best_probe_idxs = probe_wise_entropy>=top_5_lower_bound_val
            sample_pred_chosen = sample_pred[best_probe_idxs]
            sample_pred_chosen = np.argmax(sample_pred_chosen,axis=1)
            correct_answer = all_test_true[fold][0][i]
            if correct_answer==1: best_probes_nonhallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best]
            if correct_answer==1: correct_probes_nonhallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best and np.argmax(sample_pred[idx])==correct_answer]
            if correct_answer==0: best_probes_hallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best]
            if correct_answer==0: correct_probes_hallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best and np.argmax(sample_pred[idx])==correct_answer]
            if sum(sample_pred_chosen==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        fig, axs = plt.subplots(2,2)
        counts_confident_nh, bins = np.histogram(best_probes_nonhallu, bins=range(33))
        axs[0,0].stairs(counts_confident_nh, bins)
        axs[0,0].title.set_text('Non-Hallucinated')
        axs[0,0].set_ylabel('# times most confident')
        counts_confident_h, bins = np.histogram(best_probes_hallu, bins=range(33))
        axs[0,1].stairs(counts_confident_h, bins)
        axs[0,1].title.set_text('Hallucinated')
        counts, bins = np.histogram(correct_probes_nonhallu, bins=range(33))
        axs[1,0].stairs(counts*100/counts_confident_nh, bins)
        axs[1,0].set_xlabel('probe idx')
        axs[1,0].set_ylabel('# times correct (%)')
        counts, bins = np.histogram(correct_probes_hallu, bins=range(33))
        axs[1,1].stairs(counts*100/counts_confident_h, bins)
        axs[1,1].set_xlabel('probe idx')
        fig.savefig(f'{args.save_path}/figures/{args.results_file_name}_top5oracle_hist.png')
        print('Oracle (using 5 most confident):',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))

        # Oracle 3
        for top_x in [5,4,3,2,1]:
            best_sample_pred, best_sample_pred2 = [], []
            best_probe_idxs = np.argpartition(all_val_f1s[fold], -top_x)[-top_x:]
            top_5_lower_bound_val = np.min(all_val_f1s[fold][best_probe_idxs])
            val_f1_avg = np.array(val_f1_avg)
            best_probe_idxs2 = np.argpartition(val_f1_avg, -top_x)[-top_x:]
            top_5_lower_bound_val2 = np.min(val_f1_avg[best_probe_idxs2])
            print('Best probes:',best_probe_idxs)
            for i in range(all_test_pred[fold].shape[1]):
                sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
                sample_pred = sample_pred[all_val_f1s[fold]>=top_5_lower_bound_val]
                sample_pred = np.argmax(sample_pred,axis=1)
                sample_pred2 = np.squeeze(all_test_pred[fold][:,i,:])[val_f1_avg>=top_5_lower_bound_val2]
                sample_pred2 = np.argmax(sample_pred2,axis=1)
                correct_answer = all_test_true[fold][0][i]
                if sum(sample_pred==correct_answer)>0:
                    best_sample_pred.append(correct_answer)
                else:
                    best_sample_pred.append(1 if correct_answer==0 else 0)
                if sum(sample_pred2==correct_answer)>0:
                    best_sample_pred2.append(correct_answer)
                else:
                    best_sample_pred2.append(1 if correct_answer==0 else 0)
            # print('Oracle (using 5 most accurate on cls1):',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
            print('Oracle (using',str(top_x),'most accurate on both cls):',f1_score(all_test_true[fold][0],best_sample_pred2),f1_score(all_test_true[fold][0],best_sample_pred2,pos_label=0))
        
        # Oracle 1 - using logits
        print('\n')
        best_sample_pred =[]
        num_correct_probes_nonhallu, correct_probes_nonhallu = [], []
        num_correct_probes_hallu, correct_probes_hallu = [], []
        for i in range(all_test_logits[fold].shape[1]):
            sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = np.argmax(sample_pred,axis=1)
            # assert sample_pred.shape==(32,) # num_layers
            correct_answer = all_test_true[fold][0][i]
            if correct_answer==1: num_correct_probes_nonhallu.append(sum(sample_pred==correct_answer))
            if correct_answer==1 and sum(sample_pred==correct_answer)<20: correct_probes_nonhallu += [idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]
            if correct_answer==0: num_correct_probes_hallu.append(sum(sample_pred==correct_answer))
            if correct_answer==0 and sum(sample_pred==correct_answer)<20: correct_probes_hallu += [idx for idx,probe_pred in enumerate(sample_pred) if probe_pred==correct_answer]
            # if i==0: print(sample_pred==correct_answer,sum(sample_pred==correct_answer))
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        assert f1_score(all_test_true[fold][0],all_test_true[fold][0])==1
        fig, axs = plt.subplots(2,2)
        counts, bins = np.histogram(num_correct_probes_nonhallu, bins=range(33))
        axs[0,0].stairs(counts, bins)
        axs[0,0].title.set_text('Non-Hallucinated')
        axs[0,0].set_xlabel('# probes classifying correctly')
        axs[0,0].set_ylabel('# samples')
        counts, bins = np.histogram(num_correct_probes_hallu, bins=range(33))
        axs[0,1].stairs(counts, bins)
        axs[0,1].title.set_text('Hallucinated')
        axs[0,1].set_xlabel('# probes classifying correctly')
        counts, bins = np.histogram(correct_probes_nonhallu, bins=range(33))
        axs[1,0].stairs(counts, bins)
        axs[1,0].set_xlabel('probe idx')
        axs[1,0].set_ylabel('# samples correct\n(when num_correct_probes<20)')
        counts, bins = np.histogram(correct_probes_hallu, bins=range(33))
        axs[1,1].stairs(counts, bins)
        axs[1,1].set_xlabel('probe idx')
        fig.tight_layout()
        fig.savefig(f'{args.save_path}/figures/{args.results_file_name}_oracle_hist_using_logits.png')
        print('Oracle using logits:',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
        num_correct_probes_nonhallu = np.array(num_correct_probes_nonhallu)
        num_correct_probes_hallu = np.array(num_correct_probes_hallu)
        print('Non-Hallucinated hard samples:',sum(num_correct_probes_nonhallu<20),sum(num_correct_probes_nonhallu<10),sum(num_correct_probes_nonhallu<5))
        print('Hallucinated hard samples:',sum(num_correct_probes_hallu<20),sum(num_correct_probes_hallu<10),sum(num_correct_probes_hallu<5))
        print('# Hallucinated samples:',sum(all_test_true[fold][0]==0))

        # Oracle 2 - using logits
        best_sample_pred =[]
        best_probes_nonhallu, correct_probes_nonhallu = [], []
        best_probes_hallu, correct_probes_hallu = [], []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            best_probe_idxs = np.argpartition(probe_wise_entropy, -5)[-5:]
            top_5_lower_bound_val = np.min(probe_wise_entropy[best_probe_idxs])
            best_probe_idxs = probe_wise_entropy>=top_5_lower_bound_val
            sample_pred_logits = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
            sample_pred_chosen = sample_pred_logits[best_probe_idxs]
            sample_pred_chosen = np.argmax(sample_pred_chosen,axis=1)
            correct_answer = all_test_true[fold][0][i]
            if correct_answer==1: best_probes_nonhallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best]
            if correct_answer==1: correct_probes_nonhallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best and np.argmax(sample_pred_logits[idx])==correct_answer]
            if correct_answer==0: best_probes_hallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best]
            if correct_answer==0: correct_probes_hallu += [idx for idx,is_best in enumerate(best_probe_idxs) if is_best and np.argmax(sample_pred_logits[idx])==correct_answer]
            if sum(sample_pred_chosen==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        fig, axs = plt.subplots(2,2)
        counts_confident_nh, bins = np.histogram(best_probes_nonhallu, bins=range(33))
        axs[0,0].stairs(counts_confident_nh, bins)
        axs[0,0].title.set_text('Non-Hallucinated')
        axs[0,0].set_ylabel('# times most confident')
        counts_confident_h, bins = np.histogram(best_probes_hallu, bins=range(33))
        axs[0,1].stairs(counts_confident_h, bins)
        axs[0,1].title.set_text('Hallucinated')
        counts, bins = np.histogram(correct_probes_nonhallu, bins=range(33))
        axs[1,0].stairs(counts*100/counts_confident_nh, bins)
        axs[1,0].set_xlabel('probe idx')
        axs[1,0].set_ylabel('# times correct (%)')
        counts, bins = np.histogram(correct_probes_hallu, bins=range(33))
        axs[1,1].stairs(counts*100/counts_confident_h, bins)
        axs[1,1].set_xlabel('probe idx')
        fig.savefig(f'{args.save_path}/figures/{args.results_file_name}_top5oracle_hist_using_logits.png')
        print('Oracle (using 5 most confident):',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))

        # Oracle 3 - using logits
        val_f1_using_logits = np.array(val_f1_using_logits)
        best_sample_pred, best_sample_pred2 = [], []
        best_probe_idxs = np.argpartition(val_f1_using_logits, -5)[-5:]
        top_5_lower_bound_val = np.min(val_f1_using_logits[best_probe_idxs])
        val_f1_avg_using_logits = np.array(val_f1_avg_using_logits)
        best_probe_idxs2 = np.argpartition(val_f1_avg_using_logits, -5)[-5:]
        top_5_lower_bound_val2 = np.min(val_f1_avg_using_logits[best_probe_idxs2])
        print('Best probes:',best_probe_idxs)
        for i in range(all_test_logits[fold].shape[1]):
            sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[val_f1_using_logits>=top_5_lower_bound_val]
            sample_pred = np.argmax(sample_pred,axis=1)
            sample_pred2 = np.squeeze(all_test_logits[fold][:,i,:])[val_f1_avg_using_logits>=top_5_lower_bound_val2]
            sample_pred2 = np.argmax(sample_pred2,axis=1)
            correct_answer = all_test_true[fold][0][i]
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
            if sum(sample_pred2==correct_answer)>0:
                best_sample_pred2.append(correct_answer)
            else:
                best_sample_pred2.append(1 if correct_answer==0 else 0)
        print('Oracle (using 5 most accurate for cls1) using logits:',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
        print('Oracle (using 5 most accurate for both cls) using logits:',f1_score(all_test_true[fold][0],best_sample_pred2),f1_score(all_test_true[fold][0],best_sample_pred2,pos_label=0))
        print('\n')

        # Probe selection - a
        confident_sample_pred = []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        print('Using most confident probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))

        # best_probes = np.argwhere(all_val_f1s[fold]>=np.mean(all_val_f1s[fold]))
        # print('Num of probes > avg:',len(best_probes))
        # confident_sample_pred = []
        # for i in range(all_test_pred[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)[best_probes]
        #     confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        # print('Using most confident probe per sample (best probes by f1):',f1_score(all_test_true[fold][0],confident_sample_pred))
        # best_val_loss_by_model = [np.min(model_losses) for model_losses in all_val_loss[fold]]
        # best_probes = np.argwhere(best_val_loss_by_model<=np.mean(best_val_loss_by_model))
        # print('Num of probes < avg:',len(best_probes))
        # confident_sample_pred = []
        # for i in range(all_test_pred[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)[best_probes]
        #     confident_sample_pred.append(np.argmax(sample_pred[np.argmin(probe_wise_entropy)]))
        # print('Using most confident probe per sample (best probes by loss):',f1_score(all_test_true[fold][0],confident_sample_pred))

        # Probe selection - b
        confident_sample_pred1, confident_sample_pred2 = [], []
        # print(all_test_pred[fold].shape)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            best_probe_idxs = np.argpartition(probe_wise_entropy, -5)[-5:]
            top_5_lower_bound_val = np.min(probe_wise_entropy[best_probe_idxs])
            best_probe_idxs = probe_wise_entropy>=top_5_lower_bound_val
            sample_pred_chosen = sample_pred[best_probe_idxs]
            class_1_vote_cnt = sum(np.argmax(sample_pred_chosen,axis=1))
            maj_vote = 1 if class_1_vote_cnt>2 else 0
            any_vote = 1 if class_1_vote_cnt>0 else 0
            confident_sample_pred1.append(maj_vote)
            confident_sample_pred2.append(any_vote)
        print('Voting amongst 5 most confident probes per sample:',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
        # print('Any one amongst 5 most confident probes per sample:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # Probe selection - d
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst all probes per sample:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - e
        confident_sample_pred, confident_sample_pred2 = [], []
        best_probe_idxs = np.argpartition(all_val_f1s[fold], -5)[-5:]
        top_5_lower_bound_val = np.min(all_val_f1s[fold][best_probe_idxs])
        best_probe_idxs2 = np.argpartition(val_f1_avg, -5)[-5:]
        top_5_lower_bound_val2 = np.min(val_f1_avg[best_probe_idxs2])
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[all_val_f1s[fold]>=top_5_lower_bound_val]
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
            sample_pred2 = np.squeeze(all_test_pred[fold][:,i,:])[val_f1_avg>=top_5_lower_bound_val2]
            class_1_vote_cnt = sum(np.argmax(sample_pred2,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred2.shape[0]/2) else 0
            confident_sample_pred2.append(maj_vote)
        print('Voting amongst most accurate (for cls1) 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        print('Voting amongst most accurate (for both cls) 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # Probe selection - f
        # if args.use_similarity:
        #     for sim_cutoff in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        #         confident_sample_pred1, confident_sample_pred2, selected_probes = [], [], []
        #         for i in range(all_test_pred[fold].shape[1]):
        #             sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #             best_probe_idxs = (all_test_sim[fold][:,i,0]>sim_cutoff) | (all_test_sim[fold][:,i,1]>sim_cutoff)
        #             sample_pred_chosen = sample_pred[best_probe_idxs]
        #             selected_probes.append(sum(best_probe_idxs))
        #             # method 1 - vote
        #             class_1_vote_cnt = sum(np.argmax(sample_pred_chosen,axis=1))
        #             maj_vote = 1 if class_1_vote_cnt>(sum(best_probe_idxs)/2) else 0
        #             confident_sample_pred1.append(maj_vote)
        #             # method 2 - choose most confident
        #             if sum(best_probe_idxs)==0: sample_pred_chosen = sample_pred
        #             np.seterr(divide = 'ignore') # turn off for display clarity
        #             probe_wise_entropy = (-sample_pred_chosen*np.nan_to_num(np.log2(sample_pred_chosen),neginf=0)).sum(axis=1)
        #             np.seterr(divide = 'warn')
        #             confident_sample_pred2.append(np.argmax(sample_pred_chosen[np.argmin(probe_wise_entropy)]))
        #         print('Voting amongst most similar probes per sample (>',sim_cutoff,'):',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
        #         print('Using most confident amongst most similar probes per sample (>',sim_cutoff,'):',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # Probe selection - f,g
        if args.use_similarity:
            confident_sample_pred1, confident_sample_pred2, confident_sample_pred3, confident_sample_pred4, confident_sample_pred5, confident_sample_pred6, confident_sample_pred7 = [], [], [], [], [], [], []
            for i in range(all_test_pred[fold].shape[1]):
                sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
                confident_sample_pred1.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
                confident_sample_pred2.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
                if np.max(all_test_sim[fold][:,i,0]) > np.max(all_test_sim[fold][:,i,1]):
                    confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
                else:
                    confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
                sim_wgt = np.squeeze(all_test_sim[fold][:,i,:])
                # sim_wgt[sim_wgt<0] = 0 # re-assign negative weights
                confident_sample_pred4.append(np.argmax(np.sum(sample_pred*sim_wgt,axis=0)))
                confident_sample_pred5.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*all_val_f1s[fold],axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*all_val_f1s[fold],axis=0)]))
                confident_sample_pred6.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*val_f1_cls0,axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*all_val_f1s[fold],axis=0)]))
                confident_sample_pred7.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*val_f1_avg,axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*val_f1_avg,axis=0)]))
            # print('Using most similar probe per sample (cls 0 wgts):',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
            # print('Using most similar probe per sample (cls 1 wgts):',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
            print('Using most similar probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred3),f1_score(all_test_true[fold][0],confident_sample_pred3,pos_label=0))
            print('Using similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred4),f1_score(all_test_true[fold][0],confident_sample_pred4,pos_label=0))
            print('Using accuracy (cls1 acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred5),f1_score(all_test_true[fold][0],confident_sample_pred5,pos_label=0))
            print('Using accuracy (ind cls acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred6),f1_score(all_test_true[fold][0],confident_sample_pred6,pos_label=0))
            print('Using accuracy (avg acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred7),f1_score(all_test_true[fold][0],confident_sample_pred7,pos_label=0))
        # Probe selection - h
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax([np.sum(sample_pred[:,0]*probe_wise_entropy,axis=0),np.sum(sample_pred[:,1]*probe_wise_entropy,axis=0)]))
        print('Using confidence weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - i
        confident_sample_pred, confident_sample_pred1, confident_sample_pred2 = [], [], []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            confident_sample_pred.append(np.argmax([np.sum(sample_pred[:,0]*all_val_f1s[fold],axis=0),np.sum(sample_pred[:,1]*all_val_f1s[fold],axis=0)]))
            confident_sample_pred1.append(np.argmax([np.sum(sample_pred[:,0]*val_f1_cls0,axis=0),np.sum(sample_pred[:,1]*all_val_f1s[fold],axis=0)]))
            confident_sample_pred2.append(np.argmax([np.sum(sample_pred[:,0]*val_f1_avg,axis=0),np.sum(sample_pred[:,1]*val_f1_avg,axis=0)]))
        print('Using accuracy (cls1 acc) weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        print('Using accuracy (ind cls acc) weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
        print('Using accuracy (avg acc) weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # Probe selection - j,k
        probe_wgts_cls0 = []
        probe_wgts_cls1 = []
        # Load probe weights
        for model in range(all_test_pred[fold].shape[0]):
            wgt_cls0, wgt_cls1 = get_probe_wgts(fold,model,args.results_file_name,args.save_path)
            probe_wgts_cls0.append(wgt_cls0)
            probe_wgts_cls1.append(wgt_cls1)
        # Get mean similarity of each probe to every other probe
        probe_wise_mean_sim_cls0, probe_wise_mean_sim_cls1 = [], []
        all_sim_cls0, all_sim_cls1 = [], []
        for model_idx_a in range(all_test_pred[fold].shape[0]):
            sim_cls0, sim_cls1 = [], []
            norm_weights_a0 = probe_wgts_cls0[model_idx_a] / probe_wgts_cls0[model_idx_a].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            norm_weights_a1 = probe_wgts_cls1[model_idx_a] / probe_wgts_cls1[model_idx_a].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            for model_idx_b in range(all_test_pred[fold].shape[0]):
                if model_idx_b!=model_idx_a:
                    norm_weights_b0 = probe_wgts_cls0[model_idx_b] / probe_wgts_cls0[model_idx_b].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    norm_weights_b1 = probe_wgts_cls1[model_idx_b] / probe_wgts_cls1[model_idx_b].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    sim_cls0.append(torch.sum(norm_weights_a0*norm_weights_b0).item())
                    sim_cls1.append(torch.sum(norm_weights_a1*norm_weights_b1).item())
            probe_wise_mean_sim_cls0.append(np.mean(sim_cls0))
            probe_wise_mean_sim_cls1.append(np.mean(sim_cls1))
            all_sim_cls0 += sim_cls0
            all_sim_cls1 += sim_cls1
        # Majority voting
        results,results_cls1,results_cls0 = [], [], []
        results_mc,results_mc_cls1,results_mc_cls0 = [], [], []
        params = []
        for ma_top_x in [5,10,15,20,25,32]:
            for top_x in [2,3,4,5,10,15,20]:
                confident_sample_pred1, confident_sample_pred2 = [], []
                best_probe_idxs1 = np.argpartition(probe_wise_mean_sim_cls0, top_x)[:top_x]
                best_probe_idxs2 = np.argpartition(probe_wise_mean_sim_cls1, top_x)[:top_x]
                best_probe_idxs3 = np.argpartition(val_f1_avg, -ma_top_x)[-ma_top_x:]
                # best_probes_idxs_union = set(best_probe_idxs1).union(set(best_probe_idxs2))
                best_probes_idxs_union = set(best_probe_idxs2)
                best_probes_idxs = np.array(list(set.intersection(*[best_probes_idxs_union,best_probe_idxs3])),dtype=int)
                # print(best_probes_idxs)
                if len(best_probes_idxs)>0:
                    for i in range(all_test_pred[fold].shape[1]):
                        sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
                        sample_pred_chosen = sample_pred[best_probes_idxs]
                        class_1_vote_cnt = sum(np.argmax(sample_pred_chosen,axis=1))
                        maj_vote = 1 if class_1_vote_cnt>=(sample_pred_chosen.shape[0]/2) else 0
                        confident_sample_pred1.append(maj_vote)
                        probe_wise_entropy = (-sample_pred_chosen*np.nan_to_num(np.log2(sample_pred_chosen),neginf=0)).sum(axis=1)
                        confident_sample_pred2.append(np.argmax(sample_pred_chosen[np.argmin(probe_wise_entropy)]))
                    # print('Voting amongst most dissimilar probes (',ma_top_x,',',top_x,',',len(best_probes_idxs),'):',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
                    results.append(f1_score(all_test_true[fold][0],confident_sample_pred1,average='macro'))
                    results_cls1.append(f1_score(all_test_true[fold][0],confident_sample_pred1))
                    results_cls0.append(f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
                    params.append([ma_top_x,top_x,len(best_probes_idxs)])
                    results_mc.append(f1_score(all_test_true[fold][0],confident_sample_pred2,average='macro'))
                    results_mc_cls1.append(f1_score(all_test_true[fold][0],confident_sample_pred2))
                    results_mc_cls0.append(f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        best_idx = np.argmax(results)
        mc_best_idx = np.argmax(results_mc)
        print('Voting amongst most dissimilar probes (best result):',params[best_idx],results_cls1[best_idx],results_cls0[best_idx])
        print('MC amongst most dissimilar probes (best result):',params[mc_best_idx],results_cls1[mc_best_idx],results_cls0[mc_best_idx])
        # fig, axs = plt.subplots(1,2)
        # plot_data = probe_wise_mean_sim_cls0
        # counts, bins = np.histogram(plot_data)
        # axs[0,0].stairs(counts, bins)
        # plot_data = probe_wise_mean_sim_cls1
        # counts, bins = np.histogram(plot_data)
        # axs[0,1].stairs(counts, bins)
        # fig.savefig(f'{args.save_path}/figures/{args.results_file_name}_probe_avg_similarity.png')

        # Probe selection - l
        ma_top_x = 5
        confident_sample_pred, confident_sample_pred2, confident_sample_pred3 = [], [], []
        best_probe_idxs = np.argpartition(all_val_f1s[fold], -ma_top_x)[-ma_top_x:]
        top_5_lower_bound_val = np.min(all_val_f1s[fold][best_probe_idxs])
        # best_probe_idxs2 = np.argpartition(val_f1_avg, -ma_top_x)[-ma_top_x:]
        # top_5_lower_bound_val2 = np.min(val_f1_avg[best_probe_idxs2])
        # ma5_index = np.argwhere(val_f1_avg>=top_5_lower_bound_val2) # 0-31
        val_f1_cls0= np.array(val_f1_cls0)
        best_probe_idxs2 = np.argpartition(val_f1_cls0, -ma_top_x)[-ma_top_x:]
        top_5_lower_bound_val2 = np.min(val_f1_cls0[best_probe_idxs2])
        ma5_index = np.argwhere(val_f1_cls0>=top_5_lower_bound_val2) # 0-31
        ma5_index = np.array([val[0] for val in ma5_index])
        min_sim_val = 1
        for idx_a in ma5_index:
        # for idx_a in range(32):
            wgts_cls0_a, wgts_cls1_a = get_probe_wgts(fold,idx_a,args.results_file_name,args.save_path)
            # norm_weights_a = wgts_cls1_a / wgts_cls1_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            norm_weights_a = wgts_cls0_a / wgts_cls0_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            for idx_b in ma5_index:
            # for idx_b in range(32):
                if idx_b != idx_a: # for each other probe
                    wgts_cls0_b, wgts_cls1_b = get_probe_wgts(fold,idx_b,args.results_file_name,args.save_path)
                    # norm_weights_b = wgts_cls1_b / wgts_cls1_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    norm_weights_b = wgts_cls0_b / wgts_cls0_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    sim = torch.sum(norm_weights_a*norm_weights_b).item() # sim of probes
                    if sim<=min_sim_val: min_sim_val, dissimilar_idx_a, dissimilar_idx_b = sim, idx_a, idx_b
        # print(sum(val_f1_avg>=top_5_lower_bound_val2))
        num_hard_samples, entropy_wrong, entropy_correct, entropy_gap, entropy_gap_to_correct, entropy_gap2, entropy_gap_to_correct2 = 0, [], [], [], [], [], []
        max_sim, max_sim1 = [], []
        cls_wrong = []
        check_sim_correct, check_sim_wrong = [], []
        num_correct_probes= []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred_chosen = sample_pred[all_val_f1s[fold]>=top_5_lower_bound_val]
            probe_wise_entropy = (-sample_pred_chosen*np.nan_to_num(np.log2(sample_pred_chosen),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax(sample_pred_chosen[np.argmin(probe_wise_entropy)]))
            
            # sample_pred2_chosen = np.squeeze(all_test_pred[fold][:,i,:])[val_f1_avg>=top_5_lower_bound_val2]
            sample_pred2_chosen = np.squeeze(all_test_pred[fold][:,i,:])[val_f1_cls0>=top_5_lower_bound_val2]
            probe_wise_entropy = (-sample_pred2_chosen*np.nan_to_num(np.log2(sample_pred2_chosen),neginf=0)).sum(axis=1)
            confident_sample_pred2.append(np.argmax(sample_pred2_chosen[np.argmin(probe_wise_entropy)]))
            all_correct_index = np.argwhere(np.argmax(sample_pred2_chosen,axis=1)==all_test_true[fold][0][i]) # 0-4
            all_correct_index = np.array([val[0] for val in all_correct_index])
            if np.argmax(sample_pred2_chosen[np.argmin(probe_wise_entropy)])!=all_test_true[fold][0][i]:
                # print(probe_wise_entropy)
                num_hard_samples += 1
                entropy_gap.append((probe_wise_entropy[np.argpartition(probe_wise_entropy,2)[1]]-np.min(probe_wise_entropy))/np.min(probe_wise_entropy))
                if len(all_correct_index)>0: # if correct prediction exists
                    entropy_gap_to_correct.append((np.min(probe_wise_entropy[all_correct_index])-np.min(probe_wise_entropy))/np.min(probe_wise_entropy))
            else:
                num_hard_samples += 1
                entropy_gap2.append((probe_wise_entropy[np.argpartition(probe_wise_entropy,2)[1]]-np.min(probe_wise_entropy))/np.min(probe_wise_entropy))
                if len(all_correct_index)>1: # if more than 1 correct prediction exists
                    correct_index_excl_mc = [idx for idx in all_correct_index if idx!=np.argmin(probe_wise_entropy)]
                    entropy_gap_to_correct2.append((np.min(probe_wise_entropy[correct_index_excl_mc])-np.min(probe_wise_entropy))/np.min(probe_wise_entropy))
            mc_index = ma5_index[np.argmin(probe_wise_entropy)] # 0-31
            mc_wgts_cls0, mc_wgts_cls1 = get_probe_wgts(fold,mc_index,args.results_file_name,args.save_path)
            norm_weights_mc = mc_wgts_cls0 / mc_wgts_cls0.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            max_sim_val, max_sim_val1 = -1, -1
            if len(all_correct_index)>0: # if correct prediction exists
                num_correct_probes.append(len(all_correct_index))
                entropy_correct.append(np.min(probe_wise_entropy))
                for idx_a in all_correct_index: # for each correct probe
                    correct_index = ma5_index[idx_a]
                    wgts_cls0_a, wgts_cls1_a = get_probe_wgts(fold,correct_index,args.results_file_name,args.save_path)
                    # norm_weights_a = wgts_cls1_a / wgts_cls1_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    norm_weights_a = wgts_cls0_a / wgts_cls0_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    for idx_b in ma5_index:
                        if idx_b not in ma5_index[all_correct_index]: # for each incorrect probe
                            wgts_cls0_b, wgts_cls1_b = get_probe_wgts(fold,idx_b,args.results_file_name,args.save_path)
                            # norm_weights_b = wgts_cls1_b / wgts_cls1_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                            norm_weights_b = wgts_cls0_b / wgts_cls0_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                            sim = torch.sum(norm_weights_a*norm_weights_b).item() # sim of correct and incorrect probes
                            if sim>max_sim_val: max_sim_val = sim
                max_sim.append(max_sim_val)
            else:
                cls_wrong.append(all_test_true[fold][0][i])
                entropy_wrong.append(np.min(probe_wise_entropy))
                for idx_a in ma5_index:
                    wgts_cls0_a, wgts_cls1_a = get_probe_wgts(fold,idx_a,args.results_file_name,args.save_path)
                    # norm_weights_a = wgts_cls1_a / wgts_cls1_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    norm_weights_a = wgts_cls0_a / wgts_cls0_a.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    for idx_b in ma5_index:
                        if idx_b != idx_a: # for each other probe
                            wgts_cls0_b, wgts_cls1_b = get_probe_wgts(fold,idx_b,args.results_file_name,args.save_path)
                            # norm_weights_b = wgts_cls1_b / wgts_cls1_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                            norm_weights_b = wgts_cls0_b / wgts_cls0_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                            sim = torch.sum(norm_weights_a*norm_weights_b).item() # sim of probes
                            if sim>max_sim_val1: max_sim_val1 = sim
                max_sim1.append(max_sim_val1)

            # mean_probe_vector = np.mean(probe_wgts_cls0[ma5_index],axis=0)

            # sample_pred3_chosen = np.squeeze(all_test_pred[fold][:,i,:])[np.array([dissimilar_idx_a, dissimilar_idx_b])]
            # probe_wise_entropy = (-sample_pred3_chosen*np.nan_to_num(np.log2(sample_pred3_chosen),neginf=0)).sum(axis=1)
            # confident_sample_pred3.append(np.argmax(sample_pred3_chosen[np.argmin(probe_wise_entropy)]))
            # probe_wise_sim = all_test_sim[fold][:,i,0][np.array([dissimilar_idx_a, dissimilar_idx_b])]
            # confident_sample_pred3.append(np.argmax(sample_pred3_chosen[np.argmax(probe_wise_sim)]))
            # if np.argmax(sample_pred3_chosen[np.argmax(probe_wise_sim)])==all_test_true[fold][0][i]: check_sim_correct.append(np.max(probe_wise_sim))
            # if np.argmax(sample_pred3_chosen[np.argmax(probe_wise_sim)])!=all_test_true[fold][0][i]: check_sim_wrong.append(np.max(probe_wise_sim))

            # if np.max(all_test_sim[fold][:,i,0])>0.3:
            #     confident_sample_pred3.append(0)
            # else:
            #     confident_sample_pred3.append(1)
            # if all_test_true[fold][0][i]==0: check_sim_correct.append(np.max(all_test_sim[fold][:,i,1]))
            # if all_test_true[fold][0][i]==1: check_sim_wrong.append(np.max(all_test_sim[fold][:,i,1]))

            # min_sim_val = 1
            # # for idx_b in ma5_index:
            # for idx_b in range(32):
            #     if idx_b != mc_index: # for each other probe
            #         wgts_cls0_b, wgts_cls1_b = get_probe_wgts(fold,idx_b,args.results_file_name,args.save_path)
            #         # norm_weights_b = wgts_cls1_b / wgts_cls1_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            #         norm_weights_b = wgts_cls0_b / wgts_cls0_b.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
            #         sim = torch.sum(norm_weights_mc*norm_weights_b).item() # sim of probes
            #         if sim<=min_sim_val: min_sim_val, mc_dissimilar_idx = sim, idx_b
            # sample_pred3_chosen = np.squeeze(all_test_pred[fold][:,i,:])[np.array([mc_index, mc_dissimilar_idx])]
            # probe_wise_sim = all_test_sim[fold][:,i,0][np.array([mc_index, mc_dissimilar_idx])]
            # confident_sample_pred3.append(np.argmax(sample_pred3_chosen[np.argmax(probe_wise_sim)]))

            probe_wise_entropy = probe_wise_entropy[probe_wise_entropy<=0.9]
            if len(probe_wise_entropy)==0:
                confident_sample_pred3.append(0)
            else:
                confident_sample_pred3.append(np.argmax(sample_pred2_chosen[np.argmin(probe_wise_entropy)]))

        # print(len(sample_pred2_chosen))
        print('MC amongst most accurate (for cls1) 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        print('MC amongst most accurate (for both cls) 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # print(num_hard_samples)
        # fig, axs = plt.subplots(1,1)
        # counts, bins = np.histogram(entropy_gap)
        print(sum(cls_wrong))
        print(np.histogram(entropy_wrong))
        print(np.histogram(entropy_correct))
        print('Entropy gaps:')
        print(np.histogram(entropy_gap))
        print(np.histogram(entropy_gap_to_correct))
        print(np.histogram(entropy_gap2))
        print(np.histogram(entropy_gap_to_correct2))
        print('Probe similarity:')
        # print(np.histogram(max_sim))
        # print(np.histogram(max_sim1))
        # print(np.histogram(all_sim_cls0))
        # print(np.histogram(all_sim_cls1))
        print('MS between most dissimilar 2 probes amongst most accurate (for both cls) 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred3),f1_score(all_test_true[fold][0],confident_sample_pred3,pos_label=0))
        # print(np.histogram(check_sim_correct))
        # print(np.histogram(check_sim_wrong))
        print('Num correct probes:')
        print(np.histogram(num_correct_probes))

        # check_sim_correct, check_sim_wrong = [], []
        # for i in range(all_val_pred[fold].shape[1]):
        #     if all_val_true[fold][0][i]==0: check_sim_correct.append(np.max(all_val_sim[fold][:,i,0]))
        #     if all_val_true[fold][0][i]==1: check_sim_wrong.append(np.max(all_val_sim[fold][:,i,0]))
        # print('Max probe similarity of each test sample:')
        # print(np.histogram(check_sim_correct))
        # print(np.histogram(check_sim_wrong))

        # probe_wgts_cls0 = [val.detach().cpu().numpy() for val in probe_wgts_cls0]
        # # print('Probe dimensions:')
        # # print(np.histogram(np.argmax(probe_wgts_cls0, axis=1)))
        # print('PCA:')
        # pca = PCA(n_components=32) # KernelPCA(n_components=100, kernel='poly') # PCA(n_components=3)
        # transformed = pca.fit_transform(probe_wgts_cls0)
        # print(transformed.shape)
        # print(np.sum(pca.explained_variance_ratio_),pca.explained_variance_ratio_)
        # transformed = torch.from_numpy(transformed)
        # all_sim_cls0, all_sim_cls1 = [], []
        # for model_idx_a in range(all_test_pred[fold].shape[0]):
        #     sim_cls0, sim_cls1 = [], []
        #     norm_weights_a0 = transformed[model_idx_a] / transformed[model_idx_a].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
        #     # norm_weights_a1 = probe_wgts_cls1[model_idx_a] / probe_wgts_cls1[model_idx_a].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
        #     for model_idx_b in range(all_test_pred[fold].shape[0]):
        #         if model_idx_b!=model_idx_a:
        #             norm_weights_b0 = transformed[model_idx_b] / transformed[model_idx_b].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
        #             # norm_weights_b1 = probe_wgts_cls1[model_idx_b] / probe_wgts_cls1[model_idx_b].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
        #             sim_cls0.append(torch.sum(norm_weights_a0*norm_weights_b0).item())
        #             # sim_cls1.append(torch.sum(norm_weights_a1*norm_weights_b1).item())
        #     all_sim_cls0 += sim_cls0
        #     # all_sim_cls1 += sim_cls1
        # print(np.histogram(all_sim_cls0))
        
        # print('\n')
        # # Probe selection - a - using logits
        # confident_sample_pred = []
        # # print(all_test_pred[fold].shape)
        # for i in range(all_test_logits[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
        #     sample_pred_logits = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     confident_sample_pred.append(np.argmax(sample_pred_logits[np.argmin(probe_wise_entropy)]))
        # print('Using most confident probe per sample - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # # Probe selection - b - using logits
        # confident_sample_pred1, confident_sample_pred2 = [], []
        # # print(all_test_pred[fold].shape)
        # for i in range(all_test_pred[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
        #     best_probe_idxs = np.argpartition(probe_wise_entropy, -5)[-5:]
        #     top_5_lower_bound_val = np.min(probe_wise_entropy[best_probe_idxs])
        #     best_probe_idxs = probe_wise_entropy>=top_5_lower_bound_val
        #     sample_pred_logits = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     sample_pred_chosen = sample_pred_logits[best_probe_idxs]
        #     class_1_vote_cnt = sum(np.argmax(sample_pred_chosen,axis=1))
        #     maj_vote = 1 if class_1_vote_cnt>2 else 0
        #     any_vote = 1 if class_1_vote_cnt>0 else 0
        #     confident_sample_pred1.append(maj_vote)
        #     confident_sample_pred2.append(any_vote)
        # print('Voting amongst 5 most confident probes per sample - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
        # # Probe selection - d - using logits
        # confident_sample_pred = []
        # for i in range(all_test_logits[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
        #     maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
        #     confident_sample_pred.append(maj_vote)
        # print('Voting amongst all probes per sample - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # # Probe selection - e - using logits
        # confident_sample_pred, confident_sample_pred2 = [], []
        # best_probe_idxs = np.argpartition(val_f1_using_logits, -5)[-5:]
        # top_5_lower_bound_val = np.min(val_f1_using_logits[best_probe_idxs])
        # best_probe_idxs2 = np.argpartition(val_f1_avg_using_logits, -5)[-5:]
        # top_5_lower_bound_val2 = np.min(val_f1_avg_using_logits[best_probe_idxs2])
        # for i in range(all_test_logits[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     sample_pred = sample_pred[val_f1_using_logits>=top_5_lower_bound_val]
        #     class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
        #     maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
        #     confident_sample_pred.append(maj_vote)
        #     sample_pred2 = np.squeeze(all_test_logits[fold][:,i,:])[val_f1_avg_using_logits>=top_5_lower_bound_val2]
        #     class_1_vote_cnt = sum(np.argmax(sample_pred2,axis=1))
        #     maj_vote = 1 if class_1_vote_cnt>=(sample_pred2.shape[0]/2) else 0
        #     confident_sample_pred2.append(maj_vote)
        # print('Voting amongst most accurate (for cls1) 5 probes - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # print('Voting amongst most accurate (for both cls) 5 probes - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # # Probe selection - f,g - using logits
        # if args.use_similarity:
        #     confident_sample_pred1, confident_sample_pred2, confident_sample_pred3, confident_sample_pred4, confident_sample_pred5, confident_sample_pred6, confident_sample_pred7 = [], [], [], [], [], [], []
        #     for i in range(all_test_logits[fold].shape[1]):
        #         sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #         # confident_sample_pred1.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
        #         # confident_sample_pred2.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
        #         if np.max(all_test_sim[fold][:,i,0]) > np.max(all_test_sim[fold][:,i,1]):
        #             confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
        #         else:
        #             confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
        #         sim_wgt = np.squeeze(all_test_sim[fold][:,i,:])
        #         sim_wgt[sim_wgt<0] = 0 # re-assign negative weights
        #         confident_sample_pred4.append(np.argmax(np.sum(sample_pred*sim_wgt,axis=0)))
        #         confident_sample_pred5.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*val_f1_using_logits,axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*val_f1_using_logits,axis=0)]))
        #         confident_sample_pred6.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*val_f1_cls0_using_logits,axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*val_f1_using_logits,axis=0)]))
        #         confident_sample_pred7.append(np.argmax([np.sum(sample_pred[:,0]*sim_wgt[:,0]*val_f1_avg_using_logits,axis=0),np.sum(sample_pred[:,1]*sim_wgt[:,1]*val_f1_avg_using_logits,axis=0)]))
        #     print('Using most similar probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred3),f1_score(all_test_true[fold][0],confident_sample_pred3,pos_label=0))
        #     print('Using similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred4),f1_score(all_test_true[fold][0],confident_sample_pred4,pos_label=0))
        #     print('Using accuracy (cls1 acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred5),f1_score(all_test_true[fold][0],confident_sample_pred5,pos_label=0))
        #     print('Using accuracy (ind cls acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred6),f1_score(all_test_true[fold][0],confident_sample_pred6,pos_label=0))
        #     print('Using accuracy (avg acc) and similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred7),f1_score(all_test_true[fold][0],confident_sample_pred7,pos_label=0))
        # # Probe selection - h - using logits
        # confident_sample_pred = []
        # for i in range(all_test_pred[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
        #     probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
        #     sample_pred_logits = np.squeeze(all_test_logits[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     confident_sample_pred.append(np.argmax([np.sum(sample_pred_logits[:,0]*probe_wise_entropy,axis=0),np.sum(sample_pred_logits[:,1]*probe_wise_entropy,axis=0)]))
        # print('Using confidence weighted voting - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # # Probe selection - i - using logits
        # confident_sample_pred, confident_sample_pred1, confident_sample_pred2 = [], [], []
        # for i in range(all_test_pred[fold].shape[1]):
        #     sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get logits of each sample across all layers of model
        #     confident_sample_pred.append(np.argmax([np.sum(sample_pred[:,0]*val_f1_using_logits,axis=0),np.sum(sample_pred[:,1]*val_f1_using_logits,axis=0)]))
        #     confident_sample_pred1.append(np.argmax([np.sum(sample_pred[:,0]*val_f1_cls0_using_logits,axis=0),np.sum(sample_pred[:,1]*val_f1_using_logits,axis=0)]))
        #     confident_sample_pred2.append(np.argmax([np.sum(sample_pred[:,0]*val_f1_avg_using_logits,axis=0),np.sum(sample_pred[:,1]*val_f1_avg_using_logits,axis=0)]))
        # print('Using accuracy (cls1 acc) weighted voting - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # print('Using accuracy (ind cls acc) weighted voting - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
        # print('Using accuracy (avg acc) weighted voting - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))

        # print('\n')
        # np.set_printoptions(precision=2)
        # if 'ah' in args.results_file_name:
        #     print('Val loss model',model,':',all_val_loss_files[-1][fold][-1],'Val F1:',"{:.2f}".format(all_val_f1s[fold][-1]),'Test F1:',"{:.2f}".format(all_test_f1s[fold][-1]))
        # else:
        #     for model in range(len(all_val_loss[fold])):
        #         print('Val loss model',model,':',all_val_loss[fold][model],'Val F1:',"{:.2f}".format(all_val_f1s[fold][model]),'Test F1:',"{:.2f}".format(all_test_f1s[fold][model]))
        # print('\n')
        # print('Val and Test f1 correlation across probes:',np.corrcoef(all_val_f1s[fold],all_test_f1s[fold])[0][1])
        # best_val_loss_by_model = [[np.min(model_losses) for model_losses in all_val_loss[fold]] for all_val_loss in all_val_loss_files] if 'ah' in args.results_file_name else [np.min(model_losses) for model_losses in all_val_loss[fold]]
        # print('Avg val loss across probes:',np.mean(best_val_loss_by_model))
        # print('\n')



if __name__ == '__main__':
    main()