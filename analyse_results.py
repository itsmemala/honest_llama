import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score
from matplotlib import pyplot as plt
import argparse

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

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
        for layer_start,layer_end in zip(args.layer_starts,args.layer_ends):
            print(layer_start)
        # all_test_f1s = np.concatenate([np.load(f'{args.save_path}/probes/{args.results_file_name}_{args.layer_start}_{args.layer_end}_test_f1.npy') for layer_start,layer_end in zip(args.layer_starts,args.layer_ends)], axis=1)
        print(all_test_f1s.shape)
        exit()

    for fold in range(len(all_test_f1s)):

        for model in range(all_test_true[fold].shape[0]):
            assert sum(all_test_true[fold][0]==all_test_true[fold][model])==len(all_test_true[fold][0]) # check all models have same batch order
            assert sum(all_val_true[fold][0]==all_val_true[fold][model])==len(all_val_true[fold][0])

        test_f1_using_logits, val_f1_using_logits = [], []
        for model in range(all_test_logits[fold].shape[0]):
            test_f1_using_logits.append(f1_score(all_test_true[fold][0],np.argmax(all_test_logits[fold][model], axis=1)))
            val_f1_using_logits.append(f1_score(all_val_true[fold][0],np.argmax(all_val_logits[fold][model], axis=1)))

        print('FOLD',fold,'RESULTS:')
        print('Average:',np.mean(all_test_f1s[fold]))
        print('Best:',all_test_f1s[fold][np.argmax(all_val_f1s[fold])],np.argmax(all_val_f1s[fold]))
        print('Average using logits:',np.mean(test_f1_using_logits))
        
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
        best_sample_pred =[]
        best_probe_idxs = np.argpartition(all_val_f1s[fold], -5)[-5:]
        top_5_lower_bound_val = np.min(all_val_f1s[fold][best_probe_idxs])
        print('Best probes:',best_probe_idxs)
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[all_val_f1s[fold]>=top_5_lower_bound_val]
            sample_pred = np.argmax(sample_pred,axis=1)
            correct_answer = all_test_true[fold][0][i]
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        print('Oracle (using 5 most accurate):',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
        
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

        # Oracle 3 - using logits
        val_f1_using_logits = np.array(val_f1_using_logits)
        best_sample_pred =[]
        best_probe_idxs = np.argpartition(val_f1_using_logits, -5)[-5:]
        top_5_lower_bound_val = np.min(val_f1_using_logits[best_probe_idxs])
        print('Best probes:',best_probe_idxs)
        for i in range(all_test_logits[fold].shape[1]):
            sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[val_f1_using_logits>=top_5_lower_bound_val]
            sample_pred = np.argmax(sample_pred,axis=1)
            correct_answer = all_test_true[fold][0][i]
            if sum(sample_pred==correct_answer)>0:
                best_sample_pred.append(correct_answer)
            else:
                best_sample_pred.append(1 if correct_answer==0 else 0)
        print('Oracle (using 5 most accurate) using logits:',f1_score(all_test_true[fold][0],best_sample_pred),f1_score(all_test_true[fold][0],best_sample_pred,pos_label=0))
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
        print('Any one amongst 5 most confident probes per sample:',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
        # Probe selection - d
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst all probes per sample:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - e
        confident_sample_pred = []
        best_probe_idxs = np.argpartition(all_val_f1s[fold], -5)[-5:]
        top_5_lower_bound_val = np.min(all_val_f1s[fold][best_probe_idxs])
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[all_val_f1s[fold]>=top_5_lower_bound_val]
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst most accurate 5 probes:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
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
            confident_sample_pred1, confident_sample_pred2, confident_sample_pred3, confident_sample_pred4 = [], [], [], []
            for i in range(all_test_pred[fold].shape[1]):
                sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
                confident_sample_pred1.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
                confident_sample_pred2.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
                if np.max(all_test_sim[fold][:,i,0]) > np.max(all_test_sim[fold][:,i,1]):
                    confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,0])]))
                else:
                    confident_sample_pred3.append(np.argmax(sample_pred[np.argmax(all_test_sim[fold][:,i,1])]))
                sim_wgt = np.squeeze(all_test_sim[fold][:,i,:])
                sim_wgt[sim_wgt<0] = 0 # re-assign negative weights
                confident_sample_pred4.append(np.argmax(np.sum(sample_pred*sim_wgt,axis=0)))
            print('Using most similar probe per sample (cls 0 wgts):',f1_score(all_test_true[fold][0],confident_sample_pred1),f1_score(all_test_true[fold][0],confident_sample_pred1,pos_label=0))
            print('Using most similar probe per sample (cls 1 wgts):',f1_score(all_test_true[fold][0],confident_sample_pred2),f1_score(all_test_true[fold][0],confident_sample_pred2,pos_label=0))
            print('Using most similar probe per sample:',f1_score(all_test_true[fold][0],confident_sample_pred3),f1_score(all_test_true[fold][0],confident_sample_pred3,pos_label=0))
            print('Using similarity weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred4),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - h
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            probe_wise_entropy = (-sample_pred*np.nan_to_num(np.log2(sample_pred),neginf=0)).sum(axis=1)
            confident_sample_pred.append(np.argmax([np.sum(sample_pred[:,0]*probe_wise_entropy,axis=0),np.sum(sample_pred[:,1]*probe_wise_entropy,axis=0)]))
        print('Using confidence weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - i
        confident_sample_pred = []
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            confident_sample_pred.append(np.argmax([np.sum(sample_pred[:,0]*all_val_f1s[fold],axis=0),np.sum(sample_pred[:,1]*all_val_f1s[fold],axis=0)]))
        print('Using accuracy weighted voting:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        
        
        print('\n')
        # Probe selection - d - using logits
        confident_sample_pred = []
        for i in range(all_test_logits[fold].shape[1]):
            sample_pred = np.squeeze(all_test_logits[fold][:,i,:]) # Get predictions of each sample across all layers of model
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst all probes per sample - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))
        # Probe selection - e - using logits
        confident_sample_pred = []
        best_probe_idxs = np.argpartition(val_f1_using_logits, -5)[-5:]
        top_5_lower_bound_val = np.min(val_f1_using_logits[best_probe_idxs])
        for i in range(all_test_pred[fold].shape[1]):
            sample_pred = np.squeeze(all_test_pred[fold][:,i,:]) # Get predictions of each sample across all layers of model
            sample_pred = sample_pred[val_f1_using_logits>=top_5_lower_bound_val]
            class_1_vote_cnt = sum(np.argmax(sample_pred,axis=1))
            maj_vote = 1 if class_1_vote_cnt>=(sample_pred.shape[0]/2) else 0
            confident_sample_pred.append(maj_vote)
        print('Voting amongst most accurate 5 probes - using logits:',f1_score(all_test_true[fold][0],confident_sample_pred),f1_score(all_test_true[fold][0],confident_sample_pred,pos_label=0))

        print('\n')
        np.set_printoptions(precision=2)
        for model in range(len(all_val_loss[fold])):
            if 'ah' in args.results_file_name and model<1023:
                continue
            print('Val loss model',model,':',all_val_loss[fold][model],'Val F1:',"{:.2f}".format(all_val_f1s[fold][model]),'Test F1:',"{:.2f}".format(all_test_f1s[fold][model]))
        print('\n')
        print('Val and Test f1 correlation across probes:',np.corrcoef(all_val_f1s[fold],all_test_f1s[fold])[0][1])
        best_val_loss_by_model = [np.min(model_losses) for model_losses in all_val_loss[fold]]
        print('Avg val loss across probes:',np.mean(best_val_loss_by_model))
        print('\n')



if __name__ == '__main__':
    main()