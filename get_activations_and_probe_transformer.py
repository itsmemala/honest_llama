import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import datetime
import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from collections import Counter
import statistics
import pickle
# import ujson as json
# import orjson as json
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, tokenized_from_file_v2, get_token_tags
from utils import My_Transformer_Layer
from losses import SupConLoss
from copy import deepcopy
import llama
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from matplotlib import pyplot as plt
import wandb

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'hl_llama_7B': 'huggyllama/llama-7b',
    'llama_2_7B': 'meta-llama/Llama-2-7b-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama_13B': 'huggyllama/llama-13b',
    'llama_30B': 'huggyllama/llama-30b',
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))
def list_of_strs(arg):
    return list(map(str, arg.split(',')))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def num_tagged_tokens(tagged_token_idxs_prompt):
    return sum([b-a+1 for a,b in tagged_token_idxs_prompt])

def combine_acts(idx,file_name,args):
    device = args.device
    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
    if args.token=='prompt_last_and_answer_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_prompt_last/{args.model_name}_{file_name}_prompt_last_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:]],dim=1)
        # print(act1.shape,act.shape)
    elif args.token=='least_likely_and_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_least_likely/{args.model_name}_{file_name}_least_likely_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:]],dim=1)
    elif args.token=='prompt_last_and_least_likely_and_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_prompt_last/{args.model_name}_{file_name}_prompt_last_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_least_likely/{args.model_name}_{file_name}_least_likely_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act3 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:],act3[:,None,:]],dim=1)
    return act

def get_best_threshold(val_true, val_preds, is_knn=False):
    best_val_perf, best_t = 0, 0.5
    thresholds = np.histogram_bin_edges(val_preds, bins='sqrt') if is_knn else [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    print(np.histogram(val_preds, bins=thresholds))
    for t in thresholds:
        val_pred_at_thres = deepcopy(val_preds) # Deep copy so as to not touch orig values
        if is_knn:
            val_pred_at_thres[val_preds<=t] = 1 # <= to ensure correct classification when dist = [-1,0]
            val_pred_at_thres[val_preds>t] = 0
        else:
            val_pred_at_thres[val_preds>t] = 1
            val_pred_at_thres[val_preds<=t] = 0
        cls1_f1 = f1_score(val_true,val_pred_at_thres)
        cls0_f1 = f1_score(val_true,val_pred_at_thres,pos_label=0)
        perf = np.mean((cls1_f1,cls0_f1))
        if perf>best_val_perf:
            best_val_perf, best_t = perf, t
    print(best_val_perf,best_t)
    return best_t

def compute_kmeans(train_outputs,train_labels,top_k=5):
    train_outputs = train_outputs.detach().cpu().numpy()
    cluster_centers, cluster_centers_labels = [], []
    # fig, ax = 
    with warnings.catch_warnings(): # we do not want to see warnings when only one cluster is formed
        warnings.simplefilter("ignore")
        for set_id in np.unique(train_labels):
            data = np.stack([train_outputs[j] for j in train_labels if j==set_id])
            # print(data.shape)
            silhouette_avg = []
            range_k = list(range(2,top_k+1,1)) # [top_k]
            for num_clusters in range_k:
                kmeans = KMeansConstrained(n_clusters=num_clusters) # KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                cluster_labels = kmeans.labels_
                # if len(np.unique(cluster_labels))==1: # if we can form only one cluster then exit loop and set best_k=1
                #     break
                # else:
                silhouette_avg.append(silhouette_score(data, cluster_labels))
                # ax.plot(range_n_clusters,silhouette_avg,’bx-’)
            # if len(np.unique(cluster_labels))==1:
            #     best_k = 1
            # else:
            best_k = 1 # range_k[np.argmax(silhouette_avg)] # 1
            kmeans = KMeans(n_clusters=best_k)
            kmeans.fit(data)
            cluster_centers.append(kmeans.cluster_centers_)
            cluster_centers_labels += [set_id for j in range(best_k)]
            print('\nNum clusters:',len(kmeans.cluster_centers_))
    cluster_centers = np.concatenate(cluster_centers, axis=0)
    # print(cluster_centers.shape)
    # sys.exit()
    return cluster_centers, cluster_centers_labels

def compute_knn_dist(outputs,train_outputs,device,train_labels=None,metric='euclidean',top_k=5,cluster_centers=None,cluster_centers_labels=None,pca=None):
    if pca is not None:
        outputs = outputs.detach().cpu().numpy()
        outputs = torch.from_numpy(pca.transform(outputs)).to(device)
    
    dist = []
    if metric=='euclidean':
        outputs = F.normalize(outputs, p=2, dim=-1)
        train_outputs = F.normalize(train_outputs, p=2, dim=-1)
        for o in outputs:
            o_dist = torch.cdist(o[None,:], train_outputs, p=2.0)[0] # L2 distance to training data
            dist.append(o_dist[torch.argsort(o_dist)[top_k-1]]) # choose top-k sorted in ascending order (i.e. top-k smallest distances)
        dist = torch.stack(dist)
    if metric=='euclidean_avg':
        outputs = F.normalize(outputs, p=2, dim=-1)
        train_outputs = F.normalize(train_outputs, p=2, dim=-1)
        for o in outputs:
            o_dist = torch.cdist(o[None,:], train_outputs, p=2.0)[0] # L2 distance to training data
            dist.append(torch.mean(o_dist[torch.argsort(o_dist)[:top_k]])) # choose top-k sorted in ascending order (i.e. top-k smallest distances)
        dist = torch.stack(dist)
    elif metric=='euclidean_wgtd_centers' or metric=='euclidean_maj_centers':
        cluster_centers = torch.from_numpy(cluster_centers).to(device)
        for o in outputs:
            o_dist = torch.cdist(o[None,:], cluster_centers, p=2.0)[0] # L2 distance to cluster centers of training data
            cur_sample_label = cluster_centers_labels[torch.argmin(o_dist)]
            # prob_score = 1 if cur_sample_label==1 else 1e-7
            # dist.append(1 / prob_score)
            prob_score = cur_sample_label
            dist.append(-1 * prob_score)
        dist = torch.Tensor(dist)
    elif metric=='euclidean_centers':
        cluster_centers = torch.from_numpy(cluster_centers).to(device)
        for o in outputs:
            o_dist = torch.cdist(o[None,:], cluster_centers, p=2.0)[0]  # L2 distance to cluster centers of training data
            dist.append(torch.min(o_dist))
        dist = torch.Tensor(dist)
    elif metric=='mahalanobis':
        iv = torch.linalg.pinv(torch.cov(torch.transpose(train_outputs,0,1))).detach().cpu().numpy() # we want cov of the full dataset [for cov between two obs: torch.cov(torch.stack((o,t),dim=1))]
        outputs = outputs.detach().cpu().numpy()
        train_outputs = train_outputs.detach().cpu().numpy()
        for o in outputs:
            o_dist = []
            for t in train_outputs:
                # print(o.shape, t.shape) # o,t are 1-D tensors
                # print(iv.shape) # iv is (num_features,num_features)
                o_dist.append(mahalanobis(o, t, iv))
            o_dist = np.array(o_dist)
            dist.append(o_dist[np.argsort(o_dist)[top_k-1]])
        dist = torch.Tensor(dist)
    elif metric=='mahalanobis_avg':
        iv = torch.linalg.pinv(torch.cov(torch.transpose(train_outputs,0,1))).detach().cpu().numpy() # we want cov of the full dataset [for cov between two obs: torch.cov(torch.stack((o,t),dim=1))]
        outputs = outputs.detach().cpu().numpy()
        train_outputs = train_outputs.detach().cpu().numpy()
        for o in outputs:
            o_dist = []
            for t in train_outputs:
                # print(o.shape, t.shape) # o,t are 1-D tensors
                # print(iv.shape) # iv is (num_features,num_features)
                o_dist.append(mahalanobis(o, t, iv))
            o_dist = np.array(o_dist)
            dist.append(np.mean(o_dist[np.argsort(o_dist)[:top_k]]))
        dist = torch.Tensor(dist)
    elif metric=='mahalanobis_ivfix':
        outputs = outputs.detach().cpu().numpy()
        train_outputs = train_outputs.detach().cpu().numpy()
        for o in outputs:
            o_dist = []
            for t in train_outputs:
                # print(o.shape, t.shape) # o,t are 1-D tensors
                iv = np.linalg.pinv(np.cov(np.stack((o,t),axis=1)))
                # print(iv.shape) # iv is (num_features,num_features)
                o_dist.append(mahalanobis(o, t, iv))
            o_dist = np.array(o_dist)
            dist.append(o_dist[np.argsort(o_dist)[top_k-1]])
        dist = torch.Tensor(dist)
    elif metric=='mahalanobis_wgtd' or metric=='mahalanobis_maj':
        iv = []
        for set_id in [0,1]:
            data = torch.stack([train_outputs[j] for j in train_labels if j==set_id])
            iv.append(torch.linalg.pinv(torch.cov(torch.transpose(data,0,1))).detach().cpu().numpy()) # we want cov of the full dataset [for cov between two obs: torch.cov(torch.stack((o,t),dim=1))]
        outputs = outputs.detach().cpu().numpy()
        train_outputs = train_outputs.detach().cpu().numpy()
        o_matrix = []
        for o in outputs:
            o_dist = []
            for t,l in train_outputs,train_labels:
                o_dist.append(mahalanobis(o, t, iv[l]))
            o_matrix.append(np.array(o_dist))
        o_matrix = np.stack(o_matrix) # shape: (n_test_samples, n_train_samples)
        weights = 'uniform' if 'maj' in metric else 'distance'
        knn = KNeighborsClassifier(n_neighbors = top_k, metric='precomputed',weights=weights)
        knn.fit(np.ones((train_outputs.shape[0],train_outputs.shape[0])),train_labels) # dummy but required otherwise sklearn throws err
        dist = -1 * knn.predict_proba(o_matrix)[:,1] # only positive class probs; neg sign to convert probs to dist for compatibility with values returned using other metrics
        dist = torch.from_numpy(dist)
    elif metric=='mahalanobis_wgtd_centers' or metric=='mahalanobis_maj_centers':
        iv = []
        for set_id in [0,1]:
            data = torch.stack([train_outputs[j] for j in train_labels if j==set_id])
            iv.append(torch.linalg.pinv(torch.cov(torch.transpose(data,0,1))).detach().cpu().numpy()) # we want cov of the full dataset [for cov between two obs: torch.cov(torch.stack((o,t),dim=1))]
        outputs = outputs.detach().cpu().numpy()
        # o_matrix = []
        dist = []
        for o in outputs:
            o_dist = []
            for t,l in zip(cluster_centers,cluster_centers_labels):
                o_dist.append(mahalanobis(o, t, iv[l]))
            cur_sample_label = cluster_centers_labels[np.argmin(o_dist)]
            # prob_score = 1 if cur_sample_label==1 else 1e-7
            # dist.append(1 / prob_score)
            prob_score = cur_sample_label
            dist.append(-1 * prob_score)
        #     o_matrix.append(np.array(o_dist))
        # o_matrix = np.stack(o_matrix) # shape: (n_test_samples, n_train_samples)
        # weights = 'uniform' if 'maj' in metric else 'distance'
        # knn = KNeighborsClassifier(n_neighbors = 1, metric='precomputed',weights=weights)
        # knn.fit(np.ones((cluster_centers.shape[0],cluster_centers.shape[0])),cluster_centers_labels) # dummy but required otherwise sklearn throws err
        # dist = -1 * knn.predict_proba(o_matrix)[:,1] # only positive class probs; neg sign to convert probs to dist for compatibility with values returned using other metrics
        # dist = torch.from_numpy(dist)
        dist = torch.Tensor(dist)
    elif metric=='mahalanobis_centers':
        outputs = outputs.detach().cpu().numpy()
        print(torch.cov(torch.transpose(train_outputs,0,1)).shape)
        iv = torch.linalg.pinv(torch.cov(torch.transpose(train_outputs,0,1))).detach().cpu().numpy() # we want cov of the full dataset [for cov between two obs: torch.cov(torch.stack((o,t),dim=1))]
        for o in outputs:
            o_dist = []
            for t in cluster_centers:
                o_dist.append(mahalanobis(o, t, iv))
            o_dist = np.array(o_dist)
            dist.append(np.min(o_dist))
        dist = torch.Tensor(dist)
    elif metric=='cosine':
        outputs = F.normalize(outputs, p=2, dim=-1)
        train_outputs = F.normalize(train_outputs, p=2, dim=-1)
        for o in outputs:
            o_dist = 1-F.cosine_similarity(o, train_outputs, dim=-1) # cosine distance to training data
            dist.append(o_dist[torch.argsort(o_dist)[top_k-1]]) # choose top-k sorted in ascending order (i.e. top-k smallest distances)
        dist = torch.stack(dist)
    else:
        raise ValueError('Metric not implemented.')
    return dist

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--dataset_list', type=list_of_strs, default=None)
    parser.add_argument('--train_name_list', type=list_of_strs, default=None)
    parser.add_argument('--train_labels_name_list', type=list_of_strs, default=None)
    parser.add_argument('--len_dataset_list', type=list_of_ints, default=None)
    parser.add_argument('--ds_start_at_list', type=list_of_ints, default=None)
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument('--max_tokens',type=int, default=25)
    parser.add_argument('--tokens_first',type=bool, default=False) # Specifies order of tokens and layers when using_act='tagged_tokens'
    parser.add_argument('--no_sep',type=bool, default=False)
    parser.add_argument('--use_pe',type=bool, default=False)
    parser.add_argument('--method',type=str, default='transformer_hallu_pos') # (<_hallu_pos>)
    parser.add_argument('--use_layers_list',type=list_of_ints, default=None)
    parser.add_argument('--filt_prompts_catg_list',type=list_of_ints, default=None)
    parser.add_argument('--retrain_model_path',type=str, default=None)
    parser.add_argument('--retrain_full_model_path',type=str, default=None)
    parser.add_argument('--use_dropout',type=bool, default=False)
    parser.add_argument('--use_batch_norm',type=bool, default=False)
    parser.add_argument('--no_bias',type=bool, default=False)
    parser.add_argument('--norm_emb',type=bool, default=False)
    parser.add_argument('--norm_cfr',type=bool, default=False)
    parser.add_argument('--cfr_no_bias',type=bool, default=False)
    parser.add_argument('--tfr_d_model',type=int, default=128)
    parser.add_argument('--no_act_proj',type=bool, default=False)
    parser.add_argument('--norm_input',type=bool, default=False)
    parser.add_argument('--supcon_temp',type=float, default=0.1)
    parser.add_argument('--sc1_wgt',type=float, default=1)
    parser.add_argument('--sc2_wgt',type=float, default=1)
    parser.add_argument('--top_k',type=int, default=5)
    parser.add_argument('--dist_metric',type=str, default='euclidean')
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_samples',type=int, default=None)
    parser.add_argument('--test_num_samples',type=int, default=None)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--epochs',type=int, default=3)
    # parser.add_argument('--lr',type=float, default=None)
    parser.add_argument('--lr_list',default=0.05,type=list_of_floats,required=False,help='(default=%(default)s)')
    # parser.add_argument('--optimizer',type=str, default='Adam')
    parser.add_argument('--scheduler',type=str, default='warmup_cosanneal')
    parser.add_argument('--best_using_auc',type=bool, default=False)
    parser.add_argument('--best_as_last',type=bool, default=False)
    parser.add_argument('--use_best_val_t',type=bool, default=False)
    parser.add_argument('--use_class_wgt',type=bool, default=False)
    parser.add_argument('--no_batch_sampling',type=bool, default=False)
    parser.add_argument('--shuffle_batch_prompts',type=bool, default=False)
    parser.add_argument('--acts_per_file',type=int, default=100)
    parser.add_argument('--save_probes',type=bool, default=False)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--train_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--ood_test', type=bool, default=False)
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--fast_mode',type=bool, default=False) # use when GPU space is free, dataset is small and using only 1 token per sample
    # parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--seed_list',default=42,type=list_of_ints,required=False,help='(default=%(default)s)')
    parser.add_argument('--top_k_list',default=None,type=list_of_ints,required=False,help='(default=%(default)s)')
    parser.add_argument('--pca_dims_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument('--supcon_temp_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument('--skip_train', type=bool, default=False)
    parser.add_argument('--which_checkpoint', type=str, default='')
    parser.add_argument('--skip_hypsearch', type=bool, default=False)
    parser.add_argument('--continue_ce', type=bool, default=False)
    parser.add_argument('--cce_lr_list',default=None,type=list_of_floats,required=False,help='(default=%(default)s)')
    parser.add_argument('--plot_name',type=str, default=None) # Wandb args
    parser.add_argument('--tag',type=str, default=None) # Wandb args
    args = parser.parse_args()
    
    # torch.set_default_dtype(torch.float16)

    print('\n\nStart time of main:',datetime.datetime.now(),'\n\n')

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    if args.model_name=='flan_33B':
        # Cache directory
        os.environ['TRANSFORMERS_CACHE'] = args.save_path+"/"+args.model_cache_dir
        # Base model
        model_name_or_path = 'huggyllama/llama-30b' # 'huggyllama/llama-7b'
        # Adapter name on HF hub or local checkpoint path.
        # adapter_path, _ = get_last_checkpoint('qlora/output/guanaco-7b')
        adapter_path = MODEL # 'timdettmers/guanaco-7b'

        tokenizer = llama.LlamaTokenizer.from_pretrained(model_name_or_path)
        # Fixing some of the early LLaMA HF conversion issues.
        tokenizer.bos_token_id = 1

        if args.load_act==True: # Only load model if we need activations on the fly
            # Load the model (use bf16 for faster inference)
            base_model = llama.LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                # load_in_4bit=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                ),
                cache_dir=args.save_path+"/"+args.model_cache_dir
            )
            model = PeftModel.from_pretrained(base_model, adapter_path, cache_dir=args.save_path+"/"+args.model_cache_dir)
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        # if args.load_act==True: # Only load model if we need activations on the fly
        #     model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        # num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name and args.using_act=='mlp' else None #TODO: update for bigger models
        num_heads = 32
    num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0 #raise ValueError("Unknown model size.")
    args.use_layers_list = np.array(args.use_layers_list) if args.use_layers_list is not None else np.array([k for k in range(num_layers)])
    device = "cuda" if args.device is None else args.device

    print("Loading prompts and model responses..")
    all_labels = []
    args.dataset_list = [args.dataset_name] if args.dataset_list is None else args.dataset_list
    args.train_name_list = [args.train_file_name] if args.train_name_list is None else args.train_name_list
    args.train_labels_name_list = [args.train_labels_file_name] if args.train_labels_name_list is None else args.train_labels_name_list
    args.len_dataset_list = [args.len_dataset] if args.len_dataset_list is None else args.len_dataset_list
    args.ds_start_at_list = [0 for k in args.dataset_list] if args.ds_start_at_list is None else args.ds_start_at_list
    for dataset_name,train_file_name,train_labels_file_name,len_dataset,ds_start_at in zip(args.dataset_list,args.train_name_list,args.train_labels_name_list,args.len_dataset_list,args.ds_start_at_list):
        args.dataset_name = dataset_name
        args.train_file_name = train_file_name
        args.train_labels_file_name = train_labels_file_name
        args.len_dataset = len_dataset
        if args.dataset_name == 'counselling':
            file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
            prompts = tokenized_mi(file_path, tokenizer)
        elif args.dataset_name == 'gsm8k' or args.dataset_name == 'strqa' or ('baseline' in args.train_file_name or 'dola' in args.train_file_name):
            num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 11 if 'gsm8k_sampled' in args.train_file_name else 9 if 'sampled' in args.train_file_name else 1
            file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer, num_samples)
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
            labels = []
            num_samples_with_no_var = 0
            all_hallu_prompts, all_nh_prompts, hetero_prompts_sum = [], [], []
            with open(file_path, 'r') as read_file:
                data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                if 'baseline' in args.train_file_name or num_samples==1:
                    if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
                    if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
                    labels.append(label)
                else:
                    sum_over_samples = 0
                    for j in range(num_samples):
                        if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i][j]==True else 0
                        if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i][j]==True else 1
                        labels.append(label)
                        sum_over_samples += label
                    if sum_over_samples==0 or sum_over_samples==num_samples: 
                        num_samples_with_no_var += 1
                        if sum_over_samples==num_samples: all_hallu_prompts.append(i)
                        if sum_over_samples==0: all_nh_prompts.append(i)
                    else:
                        hetero_prompts_sum.append(sum_over_samples)
            labels = labels[ds_start_at:ds_start_at+args.len_dataset]
            all_labels += labels
        elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name == 'tqa_gen':
            num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 11 if 'sampled' in args.train_file_name else 1
            file_path = f'{args.save_path}/responses/{args.train_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, num_samples)
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
            if 'se_labels' in args.train_labels_file_name:
                file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.train_labels_file_name}.npy'
                labels = np.load(file_path)
            else:
                labels = []
                file_path = f'{args.save_path}/responses/{args.train_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json'
                with open(file_path, 'rb') as read_file:
                    for line in read_file:
                        data = json.loads(line)
                        # for j in range(1,num_samples+1,1):
                        #     if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                        #     if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        #     labels.append(label)
                # print('\n\End time of loading:',datetime.datetime.now(),'\n\n')
                # sys.exit()
                        if 'greedy' in args.train_labels_file_name:
                            if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                            labels.append(label)
                        else:
                            for j in range(1,num_samples+1,1):
                                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                                labels.append(label)
            labels = labels[ds_start_at:ds_start_at+args.len_dataset]
            all_labels += labels
    labels = all_labels
    test_labels = []
    if args.test_file_name is None:
        test_prompts, test_labels = [], [] # No test file
    elif 'gsm8k' in args.test_file_name or 'strqa' in args.test_file_name:
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file_v2(file_path, tokenizer)
        test_labels = []
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
            if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
            test_labels.append(label)
    else:
        file_path = f'{args.save_path}/responses/{args.test_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer,args.test_num_samples)
        if 'se_labels' in args.test_labels_file_name:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.test_labels_file_name}.npy'
            test_labels = np.load(file_path)
        else:
            test_labels = []
            print(args.test_labels_file_name)
            file_path = f'{args.save_path}/responses/{args.test_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    if 'greedy' in args.test_labels_file_name:
                        if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                        if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        test_labels.append(label)
                    else:
                        for j in range(1,args.test_num_samples+1,1):
                            if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            test_labels.append(label)
    # print(len(test_labels))
    # sys.exit()

    # print(num_samples_with_no_var)
    # print(len(all_hallu_prompts),len(all_nh_prompts))
    # print(np.histogram(hetero_prompts_sum, bins=num_samples-1))
    
    hallu_cls = 1 if 'hallu_pos' in args.method else 0

    # if args.token=='tagged_tokens':
    #     tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
    #     test_tagged_token_idxs = get_token_tags(test_prompts,test_prompt_tokens)
    # else:
    #     tagged_token_idxs,test_tagged_token_idxs = [[] for i in range(len(prompts))],[[] for i in range(len(test_prompts))]
    
    # if args.dataset_name=='strqa':
    #     args.acts_per_file = 50
    # elif args.dataset_name=='gsm8k':
    #     args.acts_per_file = 20
    # else:
    #     args.acts_per_file = 100
    
    if 'strqa' in args.test_file_name:
        args.test_acts_per_file = 50
    elif 'gsm8k' in args.test_file_name:
        args.test_acts_per_file = 20
    else:
        args.test_acts_per_file = 100

    if args.num_folds==1: # Use static test data
        if args.len_dataset==1800:
            sampled_idxs = np.random.choice(np.arange(1800), size=int(1800*(1-0.2)), replace=False) 
            test_idxs = np.array([x for x in np.arange(1800) if x not in sampled_idxs]) # Sampled indexes from 1800 held-out split
            train_idxs = sampled_idxs
        else:
            test_idxs = np.arange(len(test_labels))
            train_idxs = np.arange(len(labels)) # np.arange(args.len_dataset)
    else: # n-fold CV
        fold_idxs = np.array_split(np.arange(args.len_dataset), args.num_folds)
    
    if args.fast_mode:
        # device_id, device = 0, 'cuda:0' # start with first gpu
        print("Loading acts...")
        print('\n\nStart time of loading:',datetime.datetime.now(),'\n\n')
        my_train_acts, my_test_acts = [], []
        # if args.skip_train==False:
        for dataset_name,train_file_name,len_dataset,ds_start_at in zip(args.dataset_list,args.train_name_list,args.len_dataset_list,args.ds_start_at_list):
            args.dataset_name = dataset_name
            args.train_file_name = train_file_name
            args.len_dataset = len_dataset
            if args.dataset_name=='strqa':
                args.acts_per_file = 50
            elif args.dataset_name=='gsm8k':
                args.acts_per_file = 20
            else:
                args.acts_per_file = 100
            temp_train_idxs = train_idxs if args.dataset_list is None else np.arange(ds_start_at,ds_start_at+args.len_dataset)
            act_wise_file_paths, unique_file_paths = [], []
            for idx in temp_train_idxs:
                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                acts_train_file_name = args.train_file_name.replace('plussl','plus') if ('gsm8k' in args.train_file_name) or ('strqa' in args.train_file_name) else args.train_file_name
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{acts_train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                act_wise_file_paths.append(file_path)
                if file_path not in unique_file_paths: unique_file_paths.append(file_path)
            file_wise_data = {}
            for file_path in unique_file_paths:
                # file_wise_data[file_path] = np.load(file_path,allow_pickle=True)
                # with np.load(file_path,allow_pickle=True) as my_temp_data:
                with open(file_path, "rb") as my_temp_data:
                    file_wise_data[file_path] = pickle.load(my_temp_data)
            for idx in temp_train_idxs:
                if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                    act = combine_acts(idx,args.train_file_name,args)
                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                    if args.no_sep==False:
                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                        act = torch.cat((act,sep_token), dim=1)
                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                else:
                    # try:
                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                    act = file_wise_data[act_wise_file_paths[idx]][idx%args.acts_per_file][args.use_layers_list]
                    # print(act.shape)
                    # act = act[args.use_layers_list]
                    # print(act.shape)
                    # break
                    # except torch.cuda.OutOfMemoryError:
                    #     device_id += 1
                    #     device = 'cuda:'+str(device_id) # move to next gpu when prev is filled; test data load and rest of the processing can happen on the last gpu
                    #     print('Loading on device',device_id)
                    #     act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                my_train_acts.append(act)
        my_train_acts = torch.from_numpy(np.stack(my_train_acts)).to(device)#.type(torch.float16)

        # if args.token=='tagged_tokens': my_train_acts = torch.nn.utils.rnn.pad_sequence(my_train_acts, batch_first=True)
        
        if args.test_file_name is not None:
            for idx in test_idxs:
                file_end = idx-(idx%args.test_acts_per_file)+args.test_acts_per_file # 487: 487-(87)+100
                test_dataset_name = args.test_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa')
                file_path = f'{args.save_path}/features/{args.model_name}_{test_dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file]).to(device)
                    act = combine_acts(idx,args.test_file_name,args)
                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                    if args.no_sep==False:
                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                        act = torch.cat((act,sep_token), dim=1)
                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                else:
                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file][args.use_layers_list]).to(device)
                my_test_acts.append(act)
            # if args.token=='tagged_tokens': my_test_acts = torch.nn.utils.rnn.pad_sequence(my_test_acts, batch_first=True)
        my_test_acts = torch.stack(my_test_acts)
        print('\n\nEnd time of loading:',datetime.datetime.now(),'\n\n')

    if args.multi_gpu:
        device_id += 1
        device = 'cuda:'+str(device_id) # move to next empty gpu for model processing
        print('Loading on device',device_id)

    args.pca_dims_list = [None] if args.pca_dims_list is None else args.pca_dims_list
    for dims in args.pca_dims_list:
        args.pca_dims = dims

        args.top_k_list = [args.top_k] if args.top_k_list is None else args.top_k_list
        for top_k in args.top_k_list:
            args.top_k = top_k

            for seed_itr,save_seed in enumerate(args.seed_list):
                
                if args.skip_hypsearch:
                    lr_search_list = [args.lr_list[seed_itr]] # One-to-one mapping of seed and lr
                    supcon_temp_search_list = [args.supcon_temp_list[seed_itr]] # One-to-one mapping of seed and supcon_temp
                else:
                    lr_search_list= args.lr_list
                    supcon_temp_search_list = [args.supcon_temp] if args.supcon_temp_list is None else args.supcon_temp_list

                for supcon_temp in supcon_temp_search_list:
                    args.supcon_temp = supcon_temp

                    for lr in lr_search_list:
                        args.cce_lr_list = [None] if args.cce_lr_list is None else args.cce_lr_list
                        for cce_lr in args.cce_lr_list:
                            args.lr=lr if cce_lr is None else cce_lr
                            print('Training SEED',save_seed)
                            print('Training sc temp',args.supcon_temp)
                            print('Training lr',args.lr)

                            method_concat = args.method + '_dropout' if args.use_dropout else args.method
                            method_concat = method_concat + '_no_bias' if args.no_bias else method_concat
                            method_concat = method_concat + '_' + str(args.supcon_temp) if ('supcon' in args.method) and (args.supcon_temp!=0.1) else method_concat
                            method_concat = method_concat + '_' + str(args.sc1_wgt) + '_' + str(args.sc2_wgt) if ('supconv2_pos_wp' in args.method) and (args.sc1_wgt!=1.0 or args.sc2_wgt!=1.0) else method_concat
                            method_concat = method_concat + '_cce' if args.continue_ce else method_concat
                            method_concat = method_concat + '_' + args.dist_metric + str(args.top_k) if ('knn' in args.method) or ('kmeans' in args.method) else method_concat
                            method_concat = method_concat + 'pca' + str(args.pca_dims) if args.pca_dims is not None else method_concat
                            if args.use_batch_norm: method_concat = method_concat + '_batchnorm'
                            if args.shuffle_batch_prompts: method_concat = method_concat + '_shufflebp'
                            if args.norm_emb and args.norm_cfr and args.cfr_no_bias: method_concat += '_normcfr'
                            if args.tfr_d_model!=128: method_concat += '_dmodel' + str(args.tfr_d_model)
                            if args.no_act_proj: method_concat += 'noactproj'
                            if len(args.use_layers_list)!=num_layers: method_concat += '_'+str(min(args.use_layers_list))+'_'+str(max(args.use_layers_list))+'_'+str(len(args.use_layers_list))
                            if args.filt_prompts_catg_list is not None: method_concat += '_'.join(map(str,args.filt_prompts_catg_list))

                            # Probe training
                            np.random.seed(save_seed)
                            torch.manual_seed(save_seed)
                            if torch.cuda.is_available(): torch.cuda.manual_seed(save_seed)
                            # save_seed = save_seed if save_seed!=42 else '' # for backward compat

                            if len(args.dataset_list)==1 and args.ood_test==False:
                                probes_file_name = f'T{save_seed}_/{args.model_name}_/{args.train_file_name}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                                if 'sampled' in args.test_file_name: probes_file_name = f'T{save_seed}_/{args.model_name}_/{args.train_file_name}_{args.test_file_name}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                            elif len(args.dataset_list)>1:
                                multi_name = 'multi2' if len(args.dataset_list)==2 else 'multi'
                                if 'sampledplus' in args.train_name_list[0]: multi_name += 'sampledplus'
                                probes_file_name = f'T{save_seed}_/{args.model_name}_/{multi_name}_/{test_dataset_name}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                            elif args.ood_test:
                                train_dataset_name = args.train_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa')
                                probes_file_name = f'T{save_seed}_/{args.model_name}_/ood_{train_dataset_name}/_{test_dataset_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                            plot_name_concat = 'b' if args.use_best_val_t else ''
                            plot_name_concat += 'a' if args.best_using_auc else ''
                            plot_name_concat += 'l' if args.best_as_last else ''
                            probes_file_name += plot_name_concat
                            # Create dirs if does not exist:
                            if not os.path.exists(f'{args.save_path}/probes/models/{probes_file_name}'):
                                os.makedirs(f'{args.save_path}/probes/models/{probes_file_name}', exist_ok=True)
                            if not os.path.exists(f'{args.save_path}/probes/{probes_file_name}'):
                                os.makedirs(f'{args.save_path}/probes/{probes_file_name}', exist_ok=True)

                            # Individual probes
                            all_supcon_train_loss,all_supcon1_train_loss,all_supcon2_train_loss = {}, {}, {}
                            all_train_loss, all_val_loss, all_val_auc = {}, {}, {}
                            all_val_accs, all_val_f1s = {}, {}
                            all_test_accs, all_test_f1s = {}, {}
                            all_val_preds, all_test_preds = {}, {}
                            all_y_true_val, all_y_true_test = {}, {}
                            all_val_logits, all_test_logits = {}, {}
                            all_val_sim, all_test_sim = {}, {}

                            for i in range(args.num_folds):
                                print('Training FOLD',i)
                                train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
                                test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
                                if 'sampled' in args.train_file_name:
                                    ds_prompt_start_idx = 0
                                    for dl,tn in zip(args.len_dataset_list,args.train_name_list):
                                        # num_prompts = int(len(train_idxs)/num_samples)
                                        num_samples = 9 if 'strqa' in tn else 11
                                        num_prompts = int(dl/num_samples)
                                        # train_set_idxs = train_idxs[:int(num_prompts*(1-0.2))*num_samples] # First 80%
                                        # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
                                        labels_sample_dist = []
                                        for k in range(num_prompts):
                                            cur_prompt_idx = ds_prompt_start_idx+(k*num_samples)
                                            sample_dist = sum(labels[cur_prompt_idx:cur_prompt_idx+num_samples])
                                            if sample_dist==num_samples:
                                                labels_sample_dist.append(0)
                                            elif sample_dist==0:
                                                labels_sample_dist.append(1)
                                            elif sample_dist <= int(num_samples/3):
                                                labels_sample_dist.append(2)
                                            elif sample_dist > int(2*num_samples/3):
                                                labels_sample_dist.append(3)
                                            else:
                                                labels_sample_dist.append(4)
                                        # print(Counter(labels_sample_dist))
                                        # sys.exit()
                                        if args.filt_prompts_catg_list is None:
                                            if labels_sample_dist.count(0)==1 or labels_sample_dist.count(3)==1: labels_sample_dist[labels_sample_dist.index(3)] = 0
                                            if labels_sample_dist.count(1)==1 or labels_sample_dist.count(2)==1: labels_sample_dist[labels_sample_dist.index(2)] = 1
                                            if labels_sample_dist.count(4)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                                            if labels_sample_dist.count(0)==1: labels_sample_dist[labels_sample_dist.index(4)] = 0
                                            if labels_sample_dist.count(1)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                                        prompt_idxs_filt = np.arange(num_prompts) if args.filt_prompts_catg_list is None else [k for k in np.arange(num_prompts) if labels_sample_dist[k] in args.filt_prompts_catg_list]
                                        labels_sample_dist_filt = labels_sample_dist if args.filt_prompts_catg_list is None else [labels_sample_dist[k] for k in np.arange(num_prompts) if labels_sample_dist[k] in args.filt_prompts_catg_list]
                                        print("\n\nUsing a total of ",len(prompt_idxs_filt)," prompts.\n\n")
                                        train_prompt_idxs, val_prompt_idxs, _, _ = train_test_split(prompt_idxs_filt, labels_sample_dist_filt, stratify=labels_sample_dist_filt, test_size=0.2)
                                        # train_set_idxs = np.concatenate([np.arange(k*num_samples,(k*num_samples)+num_samples,1) for k in train_prompt_idxs], axis=0)
                                        train_set_idxs = np.concatenate([np.arange(ds_prompt_start_idx+(k*num_samples),ds_prompt_start_idx+(k*num_samples)+num_samples,1) for k in train_prompt_idxs], axis=0)
                                        # val_set_idxs = np.concatenate([np.arange(k*num_samples,(k*num_samples)+num_samples,1) for k in val_prompt_idxs], axis=0)
                                        # val_set_idxs = np.array([k*num_samples for k in val_prompt_idxs])
                                        val_set_idxs = np.array([ds_prompt_start_idx+(k*num_samples) for k in val_prompt_idxs])
                                        # assert len(train_set_idxs) + len(val_set_idxs) == args.len_dataset
                                        print('Hallu in val:',sum([labels[i] for i in val_set_idxs])/len(val_set_idxs),'Hallu in train:',sum([labels[i] for i in train_set_idxs])/len(train_set_idxs))
                                else:
                                    # train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
                                    # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
                                    train_set_idxs, val_set_idxs, _, _ = train_test_split(train_idxs, labels, stratify=labels,test_size=0.2)

                                y_train_supcon = np.stack([labels[i] for i in train_set_idxs], axis = 0)
                                y_train = np.stack([[labels[i]] for i in train_set_idxs], axis = 0)
                                y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
                                if args.test_file_name is not None: y_test = np.stack([[labels[i]] for i in test_idxs], axis = 0) if args.num_folds>1 else np.stack([test_labels[i] for i in test_idxs], axis = 0)
                                
                                all_supcon_train_loss[i], all_supcon1_train_loss[i], all_supcon2_train_loss[i], all_train_loss[i], all_val_loss[i], all_val_auc[i] = [], [], [], [], [], []
                                all_val_accs[i], all_val_f1s[i] = [], []
                                all_test_accs[i], all_test_f1s[i] = [], []
                                all_val_preds[i], all_test_preds[i] = [], []
                                all_y_true_val[i], all_y_true_test[i] = [], []
                                all_val_logits[i], all_test_logits[i] = [], []
                                all_val_sim[i], all_test_sim[i] = [], []
                                model_wise_mc_sample_idxs, probes_saved = [], []
                                
                                cur_probe_train_set_idxs = train_set_idxs
                                cur_probe_y_train = np.stack([[labels[i]] for i in cur_probe_train_set_idxs], axis = 0)
                                train_target = np.stack([labels[j] for j in cur_probe_train_set_idxs], axis = 0)
                                class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                                weight = 1. / class_sample_count
                                samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                                sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) # Default: replacement=True
                                ds_train = Dataset.from_dict({"inputs_idxs": cur_probe_train_set_idxs, "labels": cur_probe_y_train}).with_format("torch")
                                ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler) if args.no_batch_sampling==False else DataLoader(ds_train, batch_size=args.bs)
                                if args.shuffle_batch_prompts:
                                    ds_train = Dataset.from_dict({"inputs_idxs": train_prompt_idxs}).with_format("torch")
                                    ds_train = DataLoader(ds_train, batch_size=int(args.bs/num_samples), shuffle=True)
                                # sampler = RandomSampler(ds_train, replacement=True) # Default: replacement=False
                                ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                                ds_val = DataLoader(ds_val, batch_size=args.bs)
                                if args.test_file_name is not None: 
                                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                                    ds_test = DataLoader(ds_test, batch_size=args.bs)

                                act_dims = 4096
                                bias = False if 'specialised' in args.method or 'orthogonal' in args.method or args.no_bias else True
                                n_blocks = 2 if 'transformer2' in args.method else 1
                                supcon = True if 'supcon' in args.method else False
                                num_layers_to_use = num_layers if args.use_layers_list is None else len(args.use_layers_list)
                                nlinear_model = My_Transformer_Layer(n_inputs=act_dims, n_layers=num_layers_to_use, n_outputs=1, bias=bias, n_blocks=n_blocks, use_pe=args.use_pe, batch_norm=args.use_batch_norm, supcon=supcon, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias, d_model=args.tfr_d_model, no_act_proj=args.no_act_proj).to(device)
                                if args.retrain_full_model_path is not None:
                                    retrain_full_model_path = f'{args.save_path}/probes/models/{args.retrain_full_model_path}_model{i}'
                                    retrain_model_state_dict = torch.load(retrain_full_model_path).state_dict()
                                    with torch.no_grad():
                                        for n,param in nlinear_model.named_parameters():
                                            param.copy_(retrain_model_state_dict[n])
                                if args.retrain_model_path is not None:
                                    retrain_model_path = f'{args.save_path}/probes/models/{args.retrain_model_path}_model{i}'
                                    retrain_model_state_dict = torch.load(retrain_model_path).state_dict()
                                    with torch.no_grad():
                                        for n,param in nlinear_model.named_parameters():
                                            if 'classifier' not in n:
                                                param.copy_(retrain_model_state_dict[n])
                                if args.continue_ce:
                                    prior_probes_file_name = probes_file_name.replace('_cce','').replace('epochs50_'+str(args.lr),'epochs500_'+str(lr))
                                    retrain_model_path = f'{args.save_path}/probes/models/{prior_probes_file_name}_model{i}'
                                    # nlinear_model = torch.load(retrain_model_path,map_location=device)
                                    retrain_model_state_dict = torch.load(retrain_model_path).state_dict()
                                    with torch.no_grad():
                                        for n,param in nlinear_model.named_parameters():
                                            if 'classifier' not in n:
                                                param.copy_(retrain_model_state_dict[n])
                                wgt_0 = np.sum(cur_probe_y_train)/len(cur_probe_y_train)
                                criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
                                use_supcon_pos = True if 'supconv2_pos' in args.method else False
                                sc_num_samples = num_samples if 'wp' in args.method else None
                                if (use_supcon_pos) and (sc_num_samples is not None):
                                    criterion_supcon1 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=use_supcon_pos,num_samples=None) # operates on greedy samples only
                                    criterion_supcon2 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=False,num_samples=sc_num_samples,bs=args.bs) # operates within prompt only
                                else:
                                    criterion_supcon = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=use_supcon_pos,num_samples=sc_num_samples,bs=args.bs) if 'supconv2' in args.method else NTXentLoss()
                                
                                if args.norm_input:
                                    for layer in range(my_train_acts.shape[1]):
                                        transform_mean, transform_std = torch.mean(torch.stack([my_train_acts[k][layer] for k in train_set_idxs]), dim=-2), torch.std(torch.stack([my_train_acts[k][layer] for k in train_set_idxs]), dim=-2)
                                        my_train_acts[:,layer,:] = (my_train_acts[:,layer,:]-transform_mean)/transform_std
                                        my_test_acts[:,layer,:] = (my_test_acts[:,layer,:]-transform_mean)/transform_std

                                if args.skip_train==False:
                                    # Training
                                    print('\n\nStart time of train:',datetime.datetime.now(),'\n\n')
                                    supcon_train_loss, supcon1_train_loss, supcon2_train_loss, train_loss, val_loss, val_auc = [], [], [], [], [], []
                                    best_val_loss, best_val_auc = torch.inf, 0
                                    best_model_state = deepcopy(nlinear_model.state_dict())
                                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                                    named_params = list(nlinear_model.named_parameters())
                                    optimizer_grouped_parameters = [
                                        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.lr},
                                        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
                                    ]
                                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters) #torch.optim.AdamW(optimizer_grouped_parameters)
                                    # optimizer = torch.optim.Adam(nlinear_model.parameters())
                                    steps_per_epoch = int(len(train_set_idxs)/args.bs)  # number of steps in an epoch
                                    warmup_period = steps_per_epoch * 5
                                    T_max = (steps_per_epoch*args.epochs) - warmup_period # args.epochs-warmup_period
                                    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_period)
                                    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
                                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1) if args.scheduler=='static' else torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_period])
                                    # if 'supcon' in args.method:
                                    #     T_max = (steps_per_epoch*0.9*args.epochs) - warmup_period # 0.9*args.epochs-warmup_period
                                    #     scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
                                    #     scheduler3 = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=10,total_iters=steps_per_epoch*0.1*args.epochs)
                                    #     scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[warmup_period,steps_per_epoch*0.9*args.epochs])
                                    for epoch in tqdm(range(args.epochs)):
                                    # for epoch in range(args.epochs):
                                        # if epoch==250: break
                                        num_samples_used, num_val_samples_used, epoch_train_loss, epoch_supcon_loss, epoch_supcon1_loss, epoch_supcon2_loss = 0, 0, 0, 0, 0, 0
                                        nlinear_model.train()
                                        for step,batch in enumerate(ds_train):
                                            optimizer.zero_grad()
                                            activations, batch_target_idxs = [], []
                                            if 'tagged_tokens' in args.token:
                                                for k,idx in enumerate(batch['inputs_idxs']):
                                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                    act = torch.load(file_path)[idx%args.acts_per_file].to(device)
                                                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                                                    # if act.shape[1] > args.max_tokens: continue # Skip inputs with large number of tokens to avoid OOM
                                                    if act.shape[1] > args.max_tokens: act = torch.cat([act[:,:args.max_tokens,:],act[:,-1:,:]],dim=1) # Truncate inputs with large number of tokens to avoid OOM
                                                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                                                    if args.no_sep==False:
                                                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                                                        act = torch.cat((act,sep_token), dim=1)
                                                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                                                    batch_target_idxs.append(k)
                                                    activations.append(act)
                                            else:
                                                batch_input_idxs = np.concatenate([np.arange(k*num_samples,(k*num_samples)+num_samples,1) for k in batch['inputs_idxs']], axis=0) if args.shuffle_batch_prompts else batch['inputs_idxs']
                                                activations = my_train_acts[batch_input_idxs].to(device)
                                                # act = my_train_acts[idx].to(device)
                                                # activations.append(act)
                                            if len(activations)==0: continue
                                            num_samples_used += len(batch_target_idxs)
                                            if 'tagged_tokens' in args.token:
                                                inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                                            else:
                                                inputs = activations
                                                # inputs = torch.stack(activations,axis=0)
                                            # if args.norm_input: inputs = F.normalize(inputs, p=2, dim=-1) #inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-2).unsqueeze(-2))/torch.std(inputs, dim=-2).unsqueeze(-2) # mean normalise
                                            if args.shuffle_batch_prompts:
                                                targets = torch.from_numpy(np.stack([[labels[k]] for k in batch_input_idxs], axis=0))
                                            else:
                                                targets = batch['labels'][np.array(batch_target_idxs)] if 'tagged_tokens' in args.token else batch['labels']
                                            if 'supcon' in args.method:
                                                # SupCon backward
                                                if args.continue_ce==False:
                                                    emb = nlinear_model.forward_upto_classifier(inputs)
                                                    norm_emb = F.normalize(emb, p=2, dim=-1)
                                                    emb_projection = nlinear_model.projection(norm_emb)
                                                    emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                                                    if 'supconv2' in args.method:
                                                        if (use_supcon_pos) and (sc_num_samples is not None):
                                                            greedy_features_index = [k for k in range(emb_projection.shape[0]) if k%num_samples==(num_samples-1)]
                                                            if 'wp_all' in args.method:
                                                                supcon1_loss = criterion_supcon1(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates on all samples
                                                            else:
                                                                supcon1_loss = criterion_supcon1(emb_projection[greedy_features_index,None,:],torch.squeeze(targets[greedy_features_index]).to(device)) # operates on greedy samples only
                                                            supcon2_loss = criterion_supcon2(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates within prompt only
                                                            supcon_loss = args.sc1_wgt*supcon1_loss + args.sc2_wgt*supcon2_loss
                                                        else:
                                                            supcon_loss = criterion_supcon(emb_projection[:,None,:],torch.squeeze(targets).to(device))
                                                    else:
                                                        logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                                                        supcon_loss = criterion_supcon(logits, torch.squeeze(targets).to(device))
                                                    # print(supcon_loss.item())
                                                    epoch_supcon_loss += supcon_loss.item()
                                                    if (use_supcon_pos) and (sc_num_samples is not None):
                                                        epoch_supcon1_loss += supcon1_loss.item()
                                                        epoch_supcon2_loss += supcon2_loss.item()
                                                    supcon_loss.backward()
                                                # supcon_train_loss.append(supcon_loss.item())
                                                # CE backward
                                                if ('knn' in args.method) or ('kmeans' in args.method):
                                                    loss = torch.Tensor([0])
                                                else:
                                                    emb = nlinear_model.forward_upto_classifier(inputs).detach()
                                                    norm_emb = F.normalize(emb, p=2, dim=-1)
                                                    outputs = nlinear_model.classifier(norm_emb) # norm before passing here?
                                                    loss = criterion(outputs, targets.to(device).float())
                                                    loss.backward()
                                            else:
                                                outputs = nlinear_model(inputs)
                                                loss = criterion(outputs, targets.to(device).float())
                                                try:
                                                    loss.backward()
                                                except torch.cuda.OutOfMemoryError:
                                                    print('Num of tokens in input:',activations[0].shape[0])
                                            optimizer.step()
                                            scheduler.step()
                                            epoch_train_loss += loss.item()
                                            # train_loss.append(loss.item())
                                            # break
                                        # scheduler.step()
                                        if 'supcon' in args.method: epoch_supcon_loss = epoch_supcon_loss/(step+1)
                                        epoch_supcon1_loss = epoch_supcon1_loss/(step+1)
                                        epoch_supcon2_loss = epoch_supcon2_loss/(step+1)
                                        epoch_train_loss = epoch_train_loss/(step+1)

                                        # Get val loss
                                        nlinear_model.eval()
                                        epoch_val_loss = 0
                                        val_preds, val_true = [], []
                                        for step,batch in enumerate(ds_val):
                                            optimizer.zero_grad()
                                            activations, batch_target_idxs = [], []
                                            if 'tagged_tokens' in args.token: 
                                                for k,idx in enumerate(batch['inputs_idxs']):
                                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                    act = torch.load(file_path)[idx%args.acts_per_file].to(device)
                                                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                                                    # if act.shape[1] > args.max_tokens: continue # Skip inputs with large number of tokens to avoid OOM
                                                    if act.shape[1] > args.max_tokens: act = torch.cat([act[:,:args.max_tokens,:],act[:,-1:,:]],dim=1) # Truncate inputs with large number of tokens to avoid OOM
                                                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                                                    if args.no_sep==False:
                                                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                                                        act = torch.cat((act,sep_token), dim=1)
                                                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                                                    batch_target_idxs.append(k)
                                                    activations.append(act)
                                            else:
                                                activations = my_train_acts[batch['inputs_idxs']].to(device)
                                                # act = my_train_acts[idx].to(device)
                                                # activations.append(act)
                                            if len(activations)==0: continue
                                            num_val_samples_used += len(batch_target_idxs)
                                            if 'tagged_tokens' in args.token:
                                                inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                                            else:
                                                inputs = activations
                                                # inputs = torch.stack(activations,axis=0)
                                            # if args.norm_input: inputs = F.normalize(inputs, p=2, dim=-1) #inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-2).unsqueeze(-2))/torch.std(inputs, dim=-2).unsqueeze(-2) # mean normalise
                                            targets = batch['labels'][np.array(batch_target_idxs)] if 'tagged_tokens' in args.token else batch['labels']
                                            if ('knn' in args.method) or ('kmeans' in args.method):
                                                outputs = nlinear_model.forward_upto_classifier(inputs)
                                                epoch_val_loss += 0
                                                if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                    train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                    train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                else:
                                                    train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                    train_labels = None
                                                train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                val_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,train_labels,args.dist_metric,args.top_k)
                                            else:
                                                outputs = nlinear_model(inputs)
                                                epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
                                                val_preds_batch = torch.sigmoid(outputs.data)
                                            val_preds += val_preds_batch.tolist()
                                            val_true += targets.tolist()
                                        epoch_val_loss = epoch_val_loss/(step+1)
                                        # epoch_val_auc = roc_auc_score(val_true, [-v for v in val_preds]) if ('knn' in args.method) or ('kmeans' in args.method) else roc_auc_score(val_true, val_preds)
                                        epoch_val_auc = roc_auc_score(val_true, val_preds)
                                        supcon_train_loss.append(epoch_supcon_loss)
                                        supcon1_train_loss.append(epoch_supcon1_loss)
                                        supcon2_train_loss.append(epoch_supcon2_loss)
                                        train_loss.append(epoch_train_loss)
                                        val_loss.append(epoch_val_loss)
                                        val_auc.append(epoch_val_auc)
                                        # print('Loss:', epoch_supcon_loss, epoch_train_loss, epoch_val_loss)
                                        # print('Samples:',num_samples_used, num_val_samples_used)
                                        # Choose best model
                                        best_model_state_using_last = deepcopy(nlinear_model.state_dict())
                                        if epoch_val_auc > best_val_auc:
                                            best_val_auc = epoch_val_auc
                                            best_model_state_using_auc = deepcopy(nlinear_model.state_dict())
                                        if epoch_val_loss < best_val_loss:
                                                best_val_loss = epoch_val_loss
                                                best_model_state_using_loss = deepcopy(nlinear_model.state_dict())
                                        if args.best_using_auc:
                                            best_model_state = best_model_state_using_auc
                                        elif args.best_as_last:
                                            best_model_state = best_model_state_using_last
                                        else:
                                            best_model_state = best_model_state_using_loss
                                        
                                        # if args.save_probes:
                                        #     probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_epoch{epoch}_model{i}'
                                        #     torch.save(nlinear_model, probe_save_path)

                                        # Early stopping
                                        # patience, min_val_loss_drop, is_not_decreasing = 5, 0.01, 0
                                        # if len(val_loss)>=patience:
                                        #     for epoch_id in range(1,patience,1):
                                        #         val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                                        #         if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                                        #     if is_not_decreasing==patience-1: break
                                    print('\n\nEnd time of train:',datetime.datetime.now(),'\n\n')
                                    all_supcon_train_loss[i].append(np.array(supcon_train_loss))
                                    all_supcon1_train_loss[i].append(np.array(supcon1_train_loss))
                                    all_supcon2_train_loss[i].append(np.array(supcon2_train_loss))
                                    all_train_loss[i].append(np.array(train_loss))
                                    all_val_loss[i].append(np.array(val_loss))
                                    all_val_auc[i].append(np.array(val_auc))
                                    
                                    # print(np.array(val_loss))
                                    if args.save_probes:
                                        nlinear_model.load_state_dict(best_model_state)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_model{i}'
                                        torch.save(nlinear_model, probe_save_path)
                                        # with open(probe_save_path,'wb') as f:
                                        #     torch.save(nlinear_model, f)
                                        probes_saved.append(probe_save_path)

                                        nlinear_model.load_state_dict(best_model_state_using_auc)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusingauc_model{i}'
                                        torch.save(nlinear_model, probe_save_path)
                                        # torch.save(nlinear_model, probe_save_path, _use_new_zipfile_serialization=False)

                                        nlinear_model.load_state_dict(best_model_state_using_last)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusinglast_model{i}'
                                        torch.save(nlinear_model, probe_save_path)

                                        nlinear_model.load_state_dict(best_model_state_using_loss)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusingloss_model{i}'
                                        torch.save(nlinear_model, probe_save_path)
                                    
                                    nlinear_model.load_state_dict(best_model_state)
                                
                                if args.skip_train:
                                    if 'knn' in args.method or 'kmeans' in args.method:
                                        prior_probes_file_name = probes_file_name.replace('knn_','').replace('kmeans_','').replace(args.dist_metric+str(args.top_k)+'_','').replace(args.dist_metric+str(args.top_k)+'pca'+str(args.pca_dims)+'_','')
                                    elif args.ood_test:
                                        prior_probes_file_name = f'T{save_seed}_/{args.model_name}_/{args.train_file_name}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                                        prior_probes_file_name += plot_name_concat
                                    elif 'sampled' in args.test_file_name:
                                        prior_probes_file_name = probes_file_name.replace(args.test_file_name+'_','')
                                    else: # multi
                                        prior_probes_file_name = probes_file_name.replace(test_dataset_name,'trivia_qa')
                                    try:
                                        if plot_name_concat not in prior_probes_file_name: prior_probes_file_name += plot_name_concat
                                        if args.which_checkpoint not in prior_probes_file_name: prior_probes_file_name += '_' + args.which_checkpoint
                                        prior_save_path = f'{args.save_path}/probes/models/{prior_probes_file_name}_model{i}'
                                        nlinear_model = torch.load(prior_save_path,map_location=device)
                                    except FileNotFoundError:
                                        prior_probes_file_name = prior_probes_file_name.replace("/","") # FOR BACKWARD COMPATIBILITY
                                        prior_save_path = f'{args.save_path}/probes/models/{prior_probes_file_name}_model{i}'
                                        nlinear_model = torch.load(prior_save_path,map_location=device)
                                    if args.which_checkpoint not in probes_file_name: probes_file_name += '_' + args.which_checkpoint
                                    probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_model{i}'
                                    torch.save(nlinear_model, probe_save_path)

                                # Val and Test performance
                                print('\n\nStart time of val and test perf:',datetime.datetime.now(),'\n\n')
                                # if args.skip_train==False:
                                pred_correct = 0
                                y_val_pred, y_val_true = [], []
                                val_preds = []
                                val_logits = []
                                val_sim = []
                                with torch.no_grad():
                                    nlinear_model.eval()
                                    for step,batch in enumerate(ds_val):
                                        activations, batch_target_idxs = [], []
                                        for k,idx in enumerate(batch['inputs_idxs']):
                                            if 'tagged_tokens' in args.token: 
                                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                act = torch.load(file_path)[idx%args.acts_per_file].to(device)
                                                # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                                                # if act.shape[1] > args.max_tokens: continue # Skip inputs with large number of tokens to avoid OOM
                                                if act.shape[1] > args.max_tokens: act = torch.cat([act[:,:args.max_tokens,:],act[:,-1:,:]],dim=1) # Truncate inputs with large number of tokens to avoid OOM
                                                if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                                                if args.no_sep==False:
                                                    sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                                                    act = torch.cat((act,sep_token), dim=1)
                                                act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                                                batch_target_idxs.append(k)
                                            else:
                                                act = my_train_acts[idx].to(device)
                                            activations.append(act)
                                        if len(activations)==0: continue
                                        if 'tagged_tokens' in args.token:
                                            inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                                        else:
                                            inputs = torch.stack(activations,axis=0)
                                        # if args.norm_input: inputs = F.normalize(inputs, p=2, dim=-1) #inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                                        # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-2).unsqueeze(-2))/torch.std(inputs, dim=-2).unsqueeze(-2) # mean normalise
                                        if ('knn' in args.method) or ('kmeans' in args.method):
                                            outputs = nlinear_model.forward_upto_classifier(inputs)
                                            # epoch_val_loss += 0
                                            if step==0:
                                                if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                    train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                    train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                else:
                                                    train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                    train_labels= np.array([1 for idx in range(len(train_inputs))])
                                                train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                if args.pca_dims is not None:
                                                    if args.pca_dims<1:
                                                        pca = PCA(n_components=args.pca_dims,svd_solver='full')
                                                    else:
                                                        pca = PCA(n_components=int(args.pca_dims))
                                                    train_outputs = train_outputs.detach().cpu().numpy()
                                                    train_outputs = torch.from_numpy(pca.fit_transform(train_outputs)).to(device)
                                                    if train_outputs.shape[1]==1:
                                                        train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                        pca = PCA(n_components=2)
                                                        train_outputs = train_outputs.detach().cpu().numpy()
                                                        train_outputs = torch.from_numpy(pca.fit_transform(train_outputs)).to(device)
                                                else:
                                                    pca = None
                                                if 'kmeans' in args.method:
                                                    cluster_centers, cluster_centers_labels = compute_kmeans(train_outputs.data,train_labels,args.top_k)
                                                else:
                                                    cluster_centers, cluster_centers_labels = None, None
                                            # nh_train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs if labels[idx]==0],axis=0) # Take all train non-hallucinations
                                            # outputs = nlinear_model.forward_upto_classifier(nh_train_inputs)
                                            # val_preds_batch = compute_knn_dist(train_outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca)
                                            # print(torch.min(val_preds_batch),torch.quantile(val_preds_batch,0.25),torch.quantile(val_preds_batch,0.5),torch.quantile(val_preds_batch,0.75),torch.quantile(val_preds_batch,0.9),torch.max(val_preds_batch),torch.mean(val_preds_batch))
                                            val_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca)
                                            # print(torch.min(val_preds_batch),torch.quantile(val_preds_batch,0.1),torch.quantile(val_preds_batch,0.25),torch.quantile(val_preds_batch,0.5),torch.quantile(val_preds_batch,0.75),torch.max(val_preds_batch),torch.mean(val_preds_batch))
                                            # sys.exit()
                                            predicted = [1 if v<0.5 else 0 for v in val_preds_batch]
                                        else:
                                            predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:,:] to add bs dimension
                                            val_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                                        y_val_pred += predicted
                                        y_val_true += batch['labels'][np.array(batch_target_idxs)].tolist() if 'tagged_tokens' in args.token else batch['labels'].tolist()
                                        val_preds.append(val_preds_batch)
                                        val_logits.append(nlinear_model(inputs))
                                val_preds = torch.cat(val_preds).cpu().numpy()
                                all_val_preds[i].append(val_preds)
                                all_y_true_val[i].append(y_val_true)
                                all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
                                all_val_logits[i].append(torch.cat(val_logits))
                                print('Val F1: ',"%.3f" % f1_score(y_val_true,y_val_pred),"%.3f" % f1_score(y_val_true,y_val_pred,pos_label=0))
                                print('Val AUROC:',"%.3f" % roc_auc_score(y_val_true, val_preds))
                                best_val_t = get_best_threshold(y_val_true, val_preds, True if ('knn' in args.method) or ('kmeans' in args.method) else False)
                                if ('knn' in args.method) or ('kmeans' in args.method):
                                    y_val_pred_opt = [1 if v<best_val_t else 0 for v in val_preds] if args.use_best_val_t else y_val_pred
                                else:
                                    y_val_pred_opt = [1 if v>best_val_t else 0 for v in val_preds] if args.use_best_val_t else y_val_pred
                                log_val_f1 = np.mean([f1_score(y_val_true,y_val_pred_opt),f1_score(y_val_true,y_val_pred_opt,pos_label=0)])
                                log_val_recall = recall_score(y_val_true,y_val_pred_opt)
                                log_val_auc = roc_auc_score(y_val_true, [-v for v in val_preds]) if ('knn' in args.method) or ('kmeans' in args.method) else roc_auc_score(y_val_true, val_preds)
                                pred_correct = 0
                                y_test_pred, y_test_true = [], []
                                test_preds = []
                                test_logits = []
                                test_sim = []
                                samples_used_idxs = []
                                if args.test_file_name is not None: 
                                    with torch.no_grad():
                                        num_test_samples_used = 0
                                        nlinear_model.eval()
                                        for step,batch in enumerate(ds_test):
                                            activations, batch_target_idxs = [], []
                                            for k,idx in enumerate(batch['inputs_idxs']):
                                                if 'tagged_tokens' in args.token: 
                                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                    act = torch.load(file_path)[idx%args.acts_per_file].to(device)
                                                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                                                    # if act.shape[1] > args.max_tokens: continue # Skip inputs with large number of tokens to avoid OOM
                                                    if act.shape[1] > args.max_tokens: act = torch.cat([act[:,:args.max_tokens,:],act[:,-1:,:]],dim=1) # Truncate inputs with large number of tokens to avoid OOM
                                                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                                                    if args.no_sep==False:
                                                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                                                        act = torch.cat((act,sep_token), dim=1)
                                                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                                                    batch_target_idxs.append(k)
                                                else:
                                                    act = my_test_acts[idx].to(device)
                                                activations.append(act)
                                            if len(activations)==0: continue
                                            num_test_samples_used += len(batch_target_idxs)
                                            samples_used_idxs += batch['inputs_idxs'][np.array(batch_target_idxs)]
                                            if 'tagged_tokens' in args.token:
                                                inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                                            else:
                                                inputs = torch.stack(activations,axis=0)
                                            # if args.norm_input: inputs = F.normalize(inputs, p=2, dim=-1) #inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-2).unsqueeze(-2))/torch.std(inputs, dim=-2).unsqueeze(-2) # mean normalise
                                            if ('knn' in args.method) or ('kmeans' in args.method):
                                                outputs = nlinear_model.forward_upto_classifier(inputs)
                                                # epoch_val_loss += 0
                                                # if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                #     train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                #     train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                # else:
                                                #     train_inputs = torch.stack([my_train_acts[idx].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                #     train_labels = None
                                                # train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                test_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca)
                                                predicted = [1 if v<0.5 else 0 for v in test_preds_batch]
                                            else:
                                                predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:,:] to add bs dimension
                                                test_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                                            y_test_pred += predicted
                                            y_test_true += batch['labels'][np.array(batch_target_idxs)].tolist() if 'tagged_tokens' in args.token else batch['labels'].tolist()
                                            test_preds.append(test_preds_batch)
                                            test_logits.append(nlinear_model(inputs))
                                    test_preds = torch.cat(test_preds).cpu().numpy()
                                    all_test_preds[i].append(test_preds)
                                    all_y_true_test[i].append(y_test_true)
                                    all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
                                    precision, recall, _ = precision_recall_curve(y_test_true, test_preds)
                                    print('AuPR:',"%.3f" % auc(recall,precision))
                                    print('F1:',"%.3f" % f1_score(y_test_true,y_test_pred),"%.3f" % f1_score(y_test_true,y_test_pred,pos_label=0))
                                    print('Recall:',"%.3f" % recall_score(y_test_true, y_test_pred))
                                    print('AuROC:',"%.3f" % roc_auc_score(y_test_true, test_preds))
                                    print('Samples:',num_test_samples_used)
                                    if ('knn' in args.method) or ('kmeans' in args.method):
                                        y_test_pred_opt = [1 if v<best_val_t else 0 for v in test_preds] if args.use_best_val_t else y_test_pred
                                    else:
                                        y_test_pred_opt = [1 if v>best_val_t else 0 for v in test_preds] if args.use_best_val_t else y_test_pred
                                    log_test_f1 = np.mean([f1_score(y_test_true,y_test_pred_opt),f1_score(y_test_true,y_test_pred_opt,pos_label=0)])
                                    log_test_recall = recall_score(y_test_true, y_test_pred_opt)
                                    log_test_auc = roc_auc_score(y_test_true, [-v for v in test_preds]) if ('knn' in args.method) or ('kmeans' in args.method) else roc_auc_score(y_test_true, test_preds)
                                    all_test_logits[i].append(torch.cat(test_logits))

                                    # # Get preds on all tokens
                                    # alltokens_preds = []
                                    # tokenmax_preds, tokenavg_preds = [], []
                                    # acts_per_file = args.acts_per_file
                                    # for i in tqdm(range(len(test_labels))):
                                    #     # Load activations
                                    #     file_end = i-(i%acts_per_file)+acts_per_file # 487: 487-(87)+100
                                    #     file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_all/{args.model_name}_{args.test_file_name}_all_layer_wise_{file_end}.pkl'
                                    #     acts_by_layer_token = torch.from_numpy(np.load(file_path,allow_pickle=True)[i%acts_per_file]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else None # torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                    #     # acts_by_layer = acts_by_layer[layer][test_answer_token_idxes[i]:]
                                    #     preds_by_token = []
                                    #     for token_num in range(acts_by_layer_token[0][test_answer_token_idxes[i]:].shape[0]):
                                    #         token_idx = test_answer_token_idxes[i] + token_num
                                    #         inputs = torch.squeeze(acts_by_layer_token[:,token_idx,:]) # inputs: (layers, act_dims)
                                    #         inputs = inputs[None,:,:] # inp[None,:,:] to add bs dimension
                                    #         preds_by_token.append(torch.sigmoid(nlinear_model(inputs).data).cpu().numpy())
                                    #     preds_by_token = np.array(preds_by_token)
                                    #     alltokens_preds.append(preds_by_token)
                                    #     tokenmax_preds.append(1 if np.max(preds_by_token)>0.5 else 0)
                                    #     tokenavg_preds.append(1 if np.mean(preds_by_token)>0.5 else 0)
                                    # alltokens_preds_arr = np.empty(len(alltokens_preds), object)                                                 
                                    # alltokens_preds_arr[:] = alltokens_preds
                                    # np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_alltokens_preds.npy',alltokens_preds_arr)
                                    # print('Test F1 using token-avg:',f1_score(test_labels,tokenavg_preds),f1_score(test_labels,tokenavg_preds,pos_label=0))
                                    # print('Test F1 using token-max:',f1_score(test_labels,tokenmax_preds),f1_score(test_labels,tokenmax_preds,pos_label=0))
                            
                            # Free up space
                            if 'knn' in args.method or 'kmeans' in args.method:
                                del train_inputs, train_outputs
                                torch.cuda.empty_cache()


                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_auc.npy', all_val_auc)
                            # all_val_loss = np.stack([np.stack(all_val_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_loss.npy', all_val_loss)
                            # all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
                            np.save(f'{args.save_path}/probes/{probes_file_name}_train_loss.npy', all_train_loss)
                            np.save(f'{args.save_path}/probes/{probes_file_name}_supcon_train_loss.npy', all_supcon_train_loss)
                            np.save(f'{args.save_path}/probes/{probes_file_name}_supcon1_train_loss.npy', all_supcon1_train_loss)
                            np.save(f'{args.save_path}/probes/{probes_file_name}_supcon2_train_loss.npy', all_supcon2_train_loss)
                            all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_pred.npy', all_val_preds)
                            all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_f1.npy', all_val_f1s)
                            all_y_true_val = np.stack([np.array(all_y_true_val[i]) for i in range(args.num_folds)])
                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_true.npy', all_y_true_val)
                            all_val_logits = np.stack([torch.stack(all_val_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
                            np.save(f'{args.save_path}/probes/{probes_file_name}_val_logits.npy', all_val_logits)
                            # all_val_sim = np.stack([np.stack(all_val_sim[i]) for i in range(args.num_folds)])
                            # np.save(f'{args.save_path}/probes/T{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_sim.npy', all_val_sim)
                            # all_test_sim = np.stack([np.stack(all_test_sim[i]) for i in range(args.num_folds)])
                            # np.save(f'{args.save_path}/probes/T{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_sim.npy', all_test_sim)
                            np.save(f'{args.save_path}/probes/{probes_file_name}_samples_used.npy', samples_used_idxs)

                            if args.test_file_name is not None:
                                all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
                                np.save(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy', all_test_preds)
                                all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
                                np.save(f'{args.save_path}/probes/{probes_file_name}_test_f1.npy', all_test_f1s)
                                all_y_true_test = np.stack([np.array(all_y_true_test[i]) for i in range(args.num_folds)])
                                np.save(f'{args.save_path}/probes/{probes_file_name}_test_true.npy', all_y_true_test)
                                all_test_logits = np.stack([torch.stack(all_test_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
                                np.save(f'{args.save_path}/probes/{probes_file_name}_test_logits.npy', all_test_logits)

                            if args.plot_name is not None:
                                # probes_file_name = f'T{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                                val_auc = np.load(f'{args.save_path}/probes/{probes_file_name}_val_auc.npy', allow_pickle=True).item()[0]
                                val_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_val_loss.npy', allow_pickle=True).item()[0]
                                train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_train_loss.npy', allow_pickle=True).item()[0]
                                try:
                                    supcon_train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()[0]
                                except (FileNotFoundError,KeyError):
                                    supcon_train_loss = []
                                if (use_supcon_pos) and (sc_num_samples is not None):
                                    supcon1_train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_supcon1_train_loss.npy', allow_pickle=True).item()[0]
                                    supcon2_train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_supcon2_train_loss.npy', allow_pickle=True).item()[0]
                                

                                # val_loss = val_loss[-1] # Last layer only
                                # train_loss = train_loss[-1] # Last layer only
                                # if len(supcon_train_loss)>0: supcon_train_loss = supcon_train_loss[-1] # Last layer only

                                if len(val_loss)==1:
                                    val_auc = val_auc[0]
                                    val_loss = val_loss[0]
                                    train_loss = train_loss[0]
                                    if len(supcon_train_loss)>0: supcon_train_loss = supcon_train_loss[0]
                                    supcon1_train_loss = supcon1_train_loss[0]
                                    supcon2_train_loss = supcon2_train_loss[0]

                                if len(val_loss)!=len(train_loss):
                                    train_loss_by_epoch = []
                                    batches = int(len(train_loss)/len(val_loss))
                                    start_at = 0
                                    for epoch in range(len(val_loss)):
                                        train_loss_by_epoch.append(sum(train_loss[start_at:(start_at+batches)]))
                                        start_at += batches
                                    train_loss = train_loss_by_epoch

                                # print(len(val_auc))
                                # print(len(val_loss))
                                # print(len(train_loss))
                                # if len(supcon_train_loss)>0: print(len(supcon_train_loss))
                                
                                plt.subplot(1, 2, 1)
                                plt.plot(val_loss, label='val_ce_loss')
                                plt.plot(train_loss, label='train_ce_loss')
                                plt.plot(supcon_train_loss, label='train_supcon_loss')
                                if (use_supcon_pos) and (sc_num_samples is not None):
                                    plt.plot(supcon1_train_loss, label='train_supcon1_loss')
                                    plt.plot(supcon2_train_loss, label='train_supcon2_loss')
                                plt.legend(loc="upper left")
                                plt.subplot(1, 2, 2)
                                plt.plot(val_auc, label='val_auc')
                                plt.legend(loc="upper left")
                                # plt.savefig(f'{args.save_path}/testfig.png')

                                sc_hyp = '_'+str(args.supcon_temp)+'_'+str(args.sc1_wgt)+'_'+str(args.sc2_wgt) if 'supcon' in args.method else ''
                                plot_name_concat = 'b' if args.use_best_val_t else ''
                                plot_name_concat += 'a' if args.best_using_auc else ''
                                plot_name_concat += 'l' if args.best_as_last else ''

                                wandb.init(
                                project="LLM-Hallu-Detection",
                                config={
                                "run_name": probes_file_name,
                                "model": args.model_name,
                                "dataset": test_dataset_name,
                                "act_type": args.using_act,
                                "token": args.token,
                                "method": args.method,
                                "bs": args.bs,
                                "lr": args.lr,
                                "tag": args.tag, #'design_choices',
                                "norm_inp": args.norm_input,
                                "with_pe": args.use_pe,
                                # "num_blocks": args.num_blocks,
                                # "wd": args.wd
                                },
                                name=str(save_seed)+'-'+args.plot_name+sc_hyp+'-'+str(args.lr)+plot_name_concat
                                )
                                tbl = wandb.Table(columns=['Val AUC', 'Val Recall', 'Val Macro-F1', 'Test AUC', 'Test Recall', 'Test Macro-F1'],
                                            data=[[log_val_auc, log_val_recall, log_val_f1, log_test_auc, log_test_recall, log_test_f1]])
                                wandb.log({'chart': plt,
                                            'metrics': tbl
                                })
                                wandb.finish()

if __name__ == '__main__':
    main()