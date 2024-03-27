import os
from tqdm import tqdm
import numpy as np
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, get_token_tags
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
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
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--uncertainty_values_file_name", type=str, default=None, help='local directory with dataset')
    args = parser.parse_args()
    
    test_labels = []
    with open(f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json', 'r') as read_file:
        for line in read_file:
            data = json.loads(line)
            test_labels.append(1 if data['rouge1_to_target']>0.3 else 0)
    
    # Set seed
    np.random.seed(42)

    if args.num_folds==1: # Use static test data
        sampled_idxs = np.random.choice(np.arange(1800), size=int(1800*(1-0.2)), replace=False) 
        test_idxs = np.array([x for x in np.arange(1800) if x not in sampled_idxs]) # Sampled indexes from 1800 held-out split
        train_idxs = sampled_idxs if args.len_dataset==1800 else np.arange(args.len_dataset)
    else: # n-fold CV
        fold_idxs = np.array_split(np.arange(args.len_dataset), args.num_folds)
    
    for i in range(args.num_folds):
        print('FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        tot = len(test_idxs)
        class_1_perc = sum([test_labels[i] for i in test_idxs])/tot
        print('majority class:',1 if class_1_perc>0.5 else 0,'(',class_1_perc,')')
        print('baseline accuracy:',max(sum([test_labels[i] for i in test_idxs])
                                  ,tot-sum([test_labels[i] for i in test_idxs])
                                  )/tot)
        print('baseline f1:',f1_score([test_labels[i] for i in test_idxs],[1 for i in test_idxs]))

        print('\nUncertainty Baselines:')
        probs = np.load(args.uncertainty_values_file_name)
        compute_entropy_with = [('test',test_idxs),('train',train_idxs)]
        for sample_set,use_samples in compute_entropy_with:
            for use_entropy_idx in [0,1]:
                thresholds = np.histogram_bin_edges(probs[use_samples,use_entropy_idx])

                pr, recall, f1 = [], [], []
                for t in thresholds:
                    pred = probs[use_samples,use_entropy_idx]<t # accepted/non-hallucinated if below threshold
                    p, r, f, _ = precision_recall_fscore_support([test_labels[i] for i in use_samples],pred)
                    pr.append(list(p))
                    recall.append(list(r))
                    f1.append(list(f))
                idx_best_f1 = np.argmax(np.array(f1)[:,1]) # threshold for best f1 for class 1
                
                threshold_pred = probs[test_idxs,use_entropy_idx]<thresholds[idx_best_f1]
                print('Computing with',sample_set,'samples and entropy idx',use_entropy_idx,':',f1_score([test_labels[i] for i in test_idxs],threshold_pred))
    

if __name__ == '__main__':
    main()