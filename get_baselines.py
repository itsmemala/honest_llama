import os
from tqdm import tqdm
import numpy as np
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, get_token_tags
import argparse
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_fscore_support, auc, roc_auc_score
from matplotlib import pyplot as plt

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(): 
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument("--train_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_se_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_se_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_uncertainty_values_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_uncertainty_values_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()
    
    train_labels, test_labels = [], []
    num_samples_with_no_var = 0
    all_hallu_prompts, all_nh_prompts, hetero_prompts, hetero_prompts_sum = [], [], [], []
    if 'strqa' in args.dataset_name or 'gsm8k' in args.dataset_name:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.train_labels_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            sum_over_samples = 0
            if 'sampled' in args.train_labels_file_name:
                for j in range(args.num_samples):
                    train_labels.append(1 if data['is_correct'][i][j]==True else 0)
                    sum_over_samples += 1 if data['is_correct'][i][j]==True else 0
                if sum_over_samples==0 or sum_over_samples==args.num_samples: 
                    num_samples_with_no_var += 1
                    if sum_over_samples==args.num_samples: all_nh_prompts.append(i) # Note: In this file, 1 denotes non-hallu
                    if sum_over_samples==0: all_hallu_prompts.append(i)
                else:
                    hetero_prompts.append(i)
                    hetero_prompts_sum.append(args.num_samples-sum_over_samples) # Note: In this file, 1 denotes non-hallu
            else:
                train_labels.append(1 if data['is_correct'][i]==True else 0)
        if args.train_se_labels_file_name is not None:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.train_se_labels_file_name}.npy'
            train_se_labels = np.load(file_path)
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.test_labels_file_name}.json', 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            test_labels.append(1 if data['is_correct'][i]==True else 0)
    else:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.train_labels_file_name}.json', 'r') as read_file:
            i=-1
            for line in read_file:
                i += 1
                data = json.loads(line)
                sum_over_samples = 0
                if 'greedy' in args.train_labels_file_name:
                    train_labels.append(1 if data['rouge1_to_target']>0.3 else 0)
                else:
                    for j in range(1,args.num_samples+1,1):
                        train_labels.append(1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0)
                        sum_over_samples += 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0
                    if sum_over_samples==0 or sum_over_samples==args.num_samples: 
                        num_samples_with_no_var += 1
                        if sum_over_samples==args.num_samples: all_nh_prompts.append(i) # Note: In this file, 1 denotes non-hallu
                        if sum_over_samples==0: all_hallu_prompts.append(i)
                    else:
                        hetero_prompts.append(i)
                        hetero_prompts_sum.append(args.num_samples-sum_over_samples) # Note: In this file, 1 denotes non-hallu
        if args.train_se_labels_file_name is not None:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.train_se_labels_file_name}.npy'
            train_se_labels = np.load(file_path)
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.test_labels_file_name}.json', 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                test_labels.append(1 if data['rouge1_to_target']>0.3 else 0)
    train_labels = train_labels[:args.len_dataset]
    
    print(num_samples_with_no_var)
    print(len(all_hallu_prompts),len(all_nh_prompts))
    if len(hetero_prompts_sum)>0: print(np.histogram(hetero_prompts_sum, bins=args.num_samples-1))
    if args.train_se_labels_file_name is not None: print(sum([train_se_labels[i] for i in all_hallu_prompts+all_nh_prompts]))
    se_scores = np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.train_uncertainty_values_file_name}_semantic_entropy_scores.npy')
    hetero_se_scores = se_scores[np.array(hetero_prompts)]
    print(hetero_se_scores.shape, len(hetero_prompts_sum.shape))


    # Set seed
    np.random.seed(42)

    if args.num_folds==1: # Use static test data
        if args.len_dataset==1800:
            sampled_idxs = np.random.choice(np.arange(1800), size=int(1800*(1-0.2)), replace=False) 
            test_idxs = np.array([x for x in np.arange(1800) if x not in sampled_idxs]) # Sampled indexes from 1800 held-out split
            train_idxs = sampled_idxs
        else:
            test_idxs = np.arange(len(test_labels))
            train_idxs = np.arange(args.len_dataset)
    else: # n-fold CV
        fold_idxs = np.array_split(np.arange(args.len_dataset), args.num_folds)
    
    for i in range(args.num_folds):
        print('FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # class_1_perc = sum([test_labels[i] for i in test_idxs])/tot
        # print('majority class:',1 if class_1_perc>0.5 else 0,'(',class_1_perc,')')
        tot = len(train_idxs)
        print('train accuracy:',max(sum([train_labels[i] for i in train_idxs])
                                  ,tot-sum([train_labels[i] for i in train_idxs])
                                  )/tot)
        tot = len(test_idxs)
        print('test accuracy:',max(sum([test_labels[i] for i in test_idxs])
                                  ,tot-sum([test_labels[i] for i in test_idxs])
                                  )/tot)
        # print('baseline f1:',f1_score([test_labels[i] for i in test_idxs],[1 for i in test_idxs]),f1_score([test_labels[i] for i in test_idxs],[1 for i in test_idxs],pos_label=0))

        print('\nUncertainty Baselines:')
        train_probs = np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.train_uncertainty_values_file_name}_uncertainty_scores.npy')
        if 'sampled' in args.train_uncertainty_values_file_name:
            train_probs_unravel = []
            for val in train_probs:
                train_probs_unravel += val.tolist()
            train_probs = np.array(train_probs_unravel)
            assert len(train_probs)==args.len_dataset
        test_probs = np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.test_uncertainty_values_file_name}_uncertainty_scores.npy')
        compute_entropy_with = [('test',test_idxs),('train',train_idxs)]
        for sample_set,use_samples in compute_entropy_with:
            for use_entropy_idx in [0,1]:
                # print(probs[use_samples,use_entropy_idx].shape,np.count_nonzero(~np.isnan(probs[use_samples,use_entropy_idx])))
                probs = train_probs if sample_set=='train' else test_probs
                labels = train_labels if sample_set=='train' else test_labels
                threshold_data = probs[use_samples,use_entropy_idx][~np.isnan(probs[use_samples,use_entropy_idx])]
                threshold_data_labels = [labels[i] for i in use_samples[~np.isnan(probs[use_samples,use_entropy_idx])]]
                thresholds = np.histogram_bin_edges(threshold_data, bins='auto')

                pr, recall, f1 = [], [], []
                for t in thresholds:
                    pred = threshold_data<t # accepted/non-hallucinated if below threshold
                    p, r, f, _ = precision_recall_fscore_support(threshold_data_labels,pred)
                    pr.append(list(p))
                    recall.append(list(r))
                    f1.append(list(f))
                idx_best_f1_cls1 = np.argmax(np.array(f1)[:,1]) # threshold for best f1 for class 1
                idx_best_f1_avg = np.argmax(np.mean(np.array(f1),axis=1)) # threshold for best averaged f1
                
                print('Computing with',sample_set,'samples and entropy idx',use_entropy_idx,':')
                threshold_pred = test_probs[test_idxs,use_entropy_idx]<thresholds[idx_best_f1_cls1]                
                print('Optimising for cls1:',f1_score([test_labels[i] for i in test_idxs],threshold_pred),f1_score([test_labels[i] for i in test_idxs],threshold_pred,pos_label=0))
                threshold_pred = test_probs[test_idxs,use_entropy_idx]<thresholds[idx_best_f1_avg]
                print('Optimising for avg:',f1_score([test_labels[i] for i in test_idxs],threshold_pred),f1_score([test_labels[i] for i in test_idxs],threshold_pred,pos_label=0))
                # Note: we load the labels above with 0 being the hallu cls
                print('Recall for cls0 (=hallu class):',recall_score([test_labels[i] for i in test_idxs],threshold_pred,pos_label=0))
                recall, pr = [r0 for r0,r1 in recall], [p0 for p0,p1 in pr]
                print('AUPR for cls0 (=hallu class):',auc(recall,pr))
                print(sum(np.isnan(test_probs[test_idxs,use_entropy_idx])))
                print('AuROC for cls0 (=hallu class):',roc_auc_score(np.array([test_labels[i] for i in test_idxs])[~np.isnan(test_probs[test_idxs,use_entropy_idx])]
                                                                    ,test_probs[test_idxs,use_entropy_idx][~np.isnan(test_probs[test_idxs,use_entropy_idx])]))
                print('NANs in test:',sum(np.isnan(test_probs[test_idxs,use_entropy_idx])))
            
        # Semantic Entropy
        # train_probs = np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.train_uncertainty_values_file_name}_semantic_entropy_scores.npy')
        # test_probs = np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.test_uncertainty_values_file_name}_semantic_entropy_scores.npy')
        # compute_entropy_with = [('test',test_idxs),('train',train_idxs)]
        # for sample_set,use_samples in compute_entropy_with:
            

if __name__ == '__main__':
    main()