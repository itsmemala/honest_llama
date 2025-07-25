import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import datetime
import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from collections import Counter
import statistics
import pickle
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, tokenized_from_file_v2, get_token_tags
from utils import My_SupCon_NonLinear_Classifier4, LogisticRegression_Torch, Att_Pool_Layer # , My_SupCon_NonLinear_Classifier, My_SupCon_NonLinear_Classifier_wProj
from losses import SupConLoss
from copy import deepcopy
import llama
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from transformers import AutoTokenizer
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
# from k_means_constrained import KMeansConstrained
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
    'flan_33B': 'timdettmers/qlora-flan-33b',
    'llama3.1_8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1_8B_Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma_2B': 'google/gemma-2b',
    'gemma_7B': 'google/gemma-7b'
}

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

def get_best_threshold(val_true, val_preds, is_knn=False):
    best_val_perf, best_t = 0, 0.5
    thresholds = np.histogram_bin_edges(val_preds, bins='sqrt') if is_knn else [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    print(np.histogram(val_preds, bins=thresholds))
    for t in thresholds:
        val_pred_at_thres = deepcopy(val_preds) # Deep copy so as to not touch orig values
        if is_knn:
            val_pred_at_thres[val_pred_at_thres<t] = 1
            val_pred_at_thres[val_pred_at_thres>=t] = 0
        else:
            val_pred_at_thres[val_pred_at_thres>t] = 1
            val_pred_at_thres[val_pred_at_thres<=t] = 0
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
                kmeans = None #KMeansConstrained(n_clusters=num_clusters) # KMeans(n_clusters=num_clusters)
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
        # print(iv)
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
    parser.add_argument('--my_multi_name', type=str, default=None)
    parser.add_argument('--multi_probe_dataset_name',type=str, default=None)
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument('--max_answer_tokens',type=int, default=20)
    parser.add_argument('--method',type=str, default='individual_non_linear_2') # individual_linear (<_orthogonal>, <_specialised>, <reverse>, <_hallu_pos>), individual_non_linear_2 (<_supcon>, <_specialised>, <reverse>, <_hallu_pos>), individual_non_linear_3 (<_specialised>, <reverse>, <_hallu_pos>)
    parser.add_argument('--retrain_full_model_path',type=str, default=None)
    parser.add_argument('--use_dropout',type=bool, default=False)
    parser.add_argument('--no_bias',type=bool, default=False)
    parser.add_argument('--norm_emb',type=bool, default=False)
    parser.add_argument('--norm_cfr',type=bool, default=False)
    parser.add_argument('--cfr_no_bias',type=bool, default=False)
    parser.add_argument('--norm_input',type=bool, default=False)
    parser.add_argument('--supcon_temp',type=float, default=0.1)
    parser.add_argument('--spl_wgt',type=float, default=1)
    parser.add_argument('--spl_knn',type=int, default=5)
    parser.add_argument('--excl_ce',type=bool, default=False)
    parser.add_argument('--top_k',type=int, default=5)
    parser.add_argument('--dist_metric',type=str, default='euclidean')
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_samples',type=int, default=None)
    parser.add_argument('--test_num_samples',type=int, default=None)
    parser.add_argument('--use_val_aug',type=bool, default=False)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--supcon_bs',type=int, default=128)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--supcon_epochs',type=int, default=10)
    parser.add_argument('--epochs',type=int, default=3)
    parser.add_argument('--supcon_lr',type=float, default=0.05)
    parser.add_argument('--lr',type=float, default=None)
    parser.add_argument('--lr_list',default=0.05,type=list_of_floats,required=False,help='(default=%(default)s)')
    # parser.add_argument('--optimizer',type=str, default='Adam')
    parser.add_argument('--scheduler',type=str, default='warmup_cosanneal')
    parser.add_argument('--best_using_auc',type=bool, default=False)
    parser.add_argument('--best_as_last',type=bool, default=False)
    parser.add_argument('--use_best_val_t',type=bool, default=False)
    parser.add_argument('--use_class_wgt',type=bool, default=False)
    parser.add_argument('--no_batch_sampling',type=bool, default=False)
    parser.add_argument('--load_act',type=bool, default=False)
    parser.add_argument('--acts_per_file',type=int, default=100)
    parser.add_argument('--save_probes',type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
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
    parser.add_argument('--skip_train', type=bool, default=False)
    parser.add_argument('--skip_train_acts', type=bool, default=False)
    parser.add_argument('--last_only', type=bool, default=False)
    # parser.add_argument('--skip_to_layer', type=int, default=None)
    parser.add_argument('--skip_to_head', type=int, default=None)
    parser.add_argument('--which_checkpoint', type=str, default='')
    parser.add_argument('--plot_name',type=str, default=None) # Wandb args
    parser.add_argument('--tag',type=str, default=None) # Wandb args
    args = parser.parse_args()

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
    elif "llama3" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if args.load_act==True:
            model = llama3.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        if args.load_act==True: # Only load model if we need activations on the fly
            model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        # num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name and args.using_act=='mlp' else None #TODO: update for bigger models
    device = "cuda"

    num_heads = 8 if 'gemma_2B' in args.model_name else 16 if 'gemma_7B' in args.model_name else 32 if 'llama3' in args.model_name else 32

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
            if args.train_labels_file_name is not None: # if 'se_labels' in args.train_labels_file_name:
                file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.train_labels_file_name}.npy'
                labels = np.load(file_path)
            else:
                labels = []
                with open(file_path, 'r') as read_file:
                    data = json.load(read_file)
                for i in range(len(data['full_input_text'])):
                    if num_samples==1:
                        if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
                        if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
                        labels.append(label)
                    else:
                        for j in range(num_samples):
                            if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i][j]==True else 0
                            if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i][j]==True else 1
                            labels.append(label)
            labels = labels[ds_start_at:ds_start_at+args.len_dataset]
            assert len(labels)==args.len_dataset
            all_labels += labels
        elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name == 'tqa_gen' or args.dataset_name in ['city_country','movie_cast','player_date_birth']:
            num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 11 if 'sampled' in args.train_file_name else 1
            file_path = f'{args.save_path}/responses/{args.train_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, num_samples)
            prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
            if 'se_labels' in args.train_labels_file_name:
                file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.train_labels_file_name}.npy'
                labels = np.load(file_path).tolist()
            else:
                labels, rouge_scores, squad_scores = [], [], []
                file_path = f'{args.save_path}/responses/{args.train_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json'
                with open(file_path, 'r') as read_file:
                    num_samples_with_no_var = 0
                    for line in read_file:
                        data = json.loads(line)
                        # for j in range(1,num_samples+1,1):
                        #     if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                        #     if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        #     labels.append(label)
                        if 'greedy' in args.train_labels_file_name:
                            if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                            # if 'hallu_pos' in args.method: label = 0 if data['squad_f1']>0.3 else 1 # pos class is hallu
                            labels.append(label)
                            # rouge_scores.append(data['rouge1_to_target'])
                            # squad_scores.append(data['squad_f1'])
                            if(len(labels))==args.len_dataset: break
                        else:
                            sum_over_samples = 0
                            for j in range(1,num_samples+1,1):
                                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                                labels.append(label)
                                sum_over_samples += label
                            if sum_over_samples==0 or sum_over_samples==num_samples: num_samples_with_no_var += 1
            labels = labels[ds_start_at:ds_start_at+args.len_dataset]
            try:
                assert len(labels)==args.len_dataset
            except AssertionError:
                print('AssertionError with:',dataset_name,len(labels),args.len_dataset)
                sys.exit()
            all_labels += labels
    labels = all_labels
    if args.test_file_name is None:
        test_prompts, test_labels = [], [] # No test file
    else:
        if 'gsm8k' in args.test_file_name or 'strqa' in args.test_file_name or 'dola' in args.test_file_name:
            file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
            test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file_v2(file_path, tokenizer, args.test_num_samples)
            test_labels = []
            with open(file_path, 'r') as read_file:
                data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                if args.test_num_samples==1:
                    if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
                    if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
                    test_labels.append(label)
                else:
                    for j in range(args.test_num_samples):
                        if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i][j]==True else 0
                        if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i][j]==True else 1
                        test_labels.append(label)
        elif 'nq_open' in args.test_file_name or 'trivia_qa' in args.test_file_name or 'city_country' in args.test_file_name or 'movie_cast' in args.test_file_name or 'player_date_birth' in args.test_file_name:
            # num_samples = args.num_samples if ('sampled' in args.test_file_name and args.num_samples is not None) else 11 if 'sampled' in args.test_file_name else 1
            file_path = f'{args.save_path}/responses/{args.test_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
            test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer,args.test_num_samples)
            if 'se_labels' in args.test_labels_file_name:
                file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.test_labels_file_name}.npy'
                test_labels = np.load(file_path)
            else:
                test_labels, test_rouge_scores, test_squad_scores = [], [], []
                file_path = f'{args.save_path}/responses/{args.test_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json'
                with open(file_path, 'r') as read_file:
                    test_num_samples_with_no_var = 0
                    for line in read_file:
                        data = json.loads(line)
                        if 'greedy' in args.test_labels_file_name:
                            if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                            # if 'hallu_pos' in args.method: label = 0 if data['squad_f1']>0.3 else 1 # pos class is hallu
                            test_labels.append(label)
                            # test_rouge_scores.append(data['rouge1_to_target'])
                            # test_squad_scores.append(data['squad_f1'])
                        else:
                            sum_over_samples = 0
                            for j in range(1,args.test_num_samples+1,1):
                                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                                test_labels.append(label)
                                sum_over_samples += label
                            if sum_over_samples==0 or sum_over_samples==10: test_num_samples_with_no_var += 1

    
    # print(np.corrcoef(rouge_scores,squad_scores))
    # print(np.corrcoef(test_rouge_scores,test_squad_scores))
    # print(num_samples_with_no_var, test_num_samples_with_no_var)

    hallu_cls = 1 if 'hallu_pos' in args.method else 0

    if args.token=='tagged_tokens':
        tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
        test_tagged_token_idxs = get_token_tags(test_prompts,test_prompt_tokens)
    else:
        tagged_token_idxs,test_tagged_token_idxs = [[] for i in range(len(prompts))],[[] for i in range(len(test_prompts))]
    
    # if args.dataset_name=='strqa':
    #     args.acts_per_file = 50
    # elif args.dataset_name=='gsm8k':
    #     args.acts_per_file = 20
    # else:
    #     args.acts_per_file = 100

    single_token_types = ['answer_last','prompt_last','maxpool_all','slt','least_likely','after_least_likely','prompt_last_onwards']

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
        device_id, device = 0, 'cuda:0' # start with first gpu
        print("Loading acts...")
        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise','layer_att_res':'layer_wise'}
        my_train_acts, my_train_acts2, my_test_acts = [], [], []
        # if args.skip_train==False:
        if args.skip_train_acts==False:
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
                print("Loading files..")
                for file_path in unique_file_paths:
                    file_wise_data[file_path] = np.load(file_path,allow_pickle=True)
                    if args.using_act=='layer_att_res':
                        file_path2 = file_path.replace('layer_wise','attresout_wise')
                        file_wise_data[file_path2] = np.load(file_path2,allow_pickle=True)
                actual_answer_width = []
                for idx in tqdm(temp_train_idxs):
                    # file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                    # file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                    # try:
                        # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                    act = file_wise_data[act_wise_file_paths[idx]][idx%args.acts_per_file]
                    if args.using_act=='layer_att_res': 
                        file_path2 = act_wise_file_paths[idx].replace('layer_wise','attresout_wise')
                        act2 = file_wise_data[file_path2][idx%args.acts_per_file]
                        act = np.concatenate([act,act2],axis=0)
                        if args.token=='prompt_last_onwards':
                            actual_answer_width.append(act.shape[1])
                            max_tokens = args.max_answer_tokens
                            if act.shape[1]<max_tokens: # Let max num of answer tokens be max_tokens
                                pads = np.zeros([act.shape[0],max_tokens-act.shape[1],act.shape[2]])
                                act = np.concatenate([act,pads],axis=1)
                            elif act.shape[1]>max_tokens:
                                act = act[:,-max_tokens:,:] # Only most recent tokens
                        # print(act.shape)
                    # except IndexError:
                    #     print(idx)
                    # except (torch.cuda.OutOfMemoryError, RuntimeError):
                    #     device_id += 1
                    #     device = 'cuda:'+str(device_id) # move to next gpu when prev is filled; test data load and rest of the processing can happen on the last gpu
                    #     print('Loading on device',device_id)
                    #     act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                    my_train_acts.append(act)
                # print(np.histogram(actual_answer_width), np.max(actual_answer_width))
            my_train_acts = torch.from_numpy(np.stack(my_train_acts)).to(device).to(torch.float64)
            print(my_train_acts.shape)
        
        if args.test_file_name is not None:
            print("Loading test acts...",len(test_labels))
            if 'strqa' in args.test_file_name:
                args.acts_per_file = 50
            elif 'gsm8k' in args.test_file_name:
                args.acts_per_file = 20
            else:
                args.acts_per_file = 100
            act_wise_file_paths, unique_file_paths = [], []
            for idx in test_idxs:
                file_end = idx-(idx%args.test_acts_per_file)+args.test_acts_per_file # 487: 487-(87)+100
                test_dataset_name = args.test_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa').replace('city','city_country').replace('movie','movie_cast').replace('player','player_date_birth')
                file_path = f'{args.save_path}/features/{args.model_name}_{test_dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                act_wise_file_paths.append(file_path)
                if file_path not in unique_file_paths: unique_file_paths.append(file_path)
            file_wise_data = {}
            for file_path in unique_file_paths:
                with open(file_path, "rb") as my_temp_data:
                    file_wise_data[file_path] = pickle.load(my_temp_data)
                if args.using_act=='layer_att_res':
                    file_path2 = file_path.replace('layer_wise','attresout_wise')
                    with open(file_path2, "rb") as my_temp_data:
                        file_wise_data[file_path2] = pickle.load(my_temp_data)
            actual_answer_width = []
            for idx in test_idxs:
                act = file_wise_data[act_wise_file_paths[idx]][idx%args.test_acts_per_file]
                if args.using_act=='layer_att_res': 
                    file_path2 = act_wise_file_paths[idx].replace('layer_wise','attresout_wise')
                    act2 = file_wise_data[file_path2][idx%args.test_acts_per_file]
                    act = np.concatenate([act,act2],axis=0)
                    if args.token=='prompt_last_onwards':
                        actual_answer_width.append(act.shape[1])
                        if act.shape[1]<max_tokens: # Let max num of answer tokens be max_tokens
                            pads = np.zeros([act.shape[0],max_tokens-act.shape[1],act.shape[2]])
                            act = np.concatenate([act,pads],axis=1)
                        elif act.shape[1]>max_tokens:
                            act = act[:,-max_tokens:,:] # Only last 50 tokens
                my_test_acts.append(act)
            # print(np.histogram(actual_answer_width), np.max(actual_answer_width))
        my_test_acts = torch.from_numpy(np.stack(my_test_acts)).to(device).to(torch.float64)

    if args.multi_gpu:
        device_id += 1
        device = 'cuda:'+str(device_id) # move to next empty gpu for model processing

    # print('\nact_dims:',my_train_acts.shape,'\n')

    args.pca_dims_list = [None] if args.pca_dims_list is None else args.pca_dims_list
    for dims in args.pca_dims_list:
        args.pca_dims = dims

        args.top_k_list = [args.top_k] if args.top_k_list is None else args.top_k_list
        for top_k in args.top_k_list:
            args.top_k = top_k

            for seed_itr,save_seed in enumerate(args.seed_list):
                print('Training SEED',save_seed)

                for lr in args.lr_list:
                    print('Training lr',lr)
                    args.lr=lr

                    method_concat = args.method + '_dropout' if args.use_dropout else args.method
                    method_concat = method_concat + '_no_bias' if args.no_bias else method_concat
                    method_concat = method_concat + '_valaug' if args.use_val_aug else method_concat
                    method_concat = method_concat + '_' + str(args.supcon_bs) + '_' + str(args.supcon_epochs) + '_' + str(args.supcon_lr) + '_' + str(args.supcon_temp) if 'supcon' in args.method else method_concat
                    method_concat = method_concat + '_' + str(args.spl_wgt) + '_' + str(args.spl_knn) if 'specialised' in args.method else method_concat
                    method_concat = method_concat + '_' + str(args.spl_wgt) if 'orthogonal' in args.method else method_concat
                    method_concat = method_concat + '_' + str(args.spl_knn) if 'orthogonal' in args.method and args.excl_ce else method_concat
                    method_concat = method_concat + '_excl_ce' if args.excl_ce else method_concat
                    method_concat = method_concat + '_' + args.dist_metric + str(args.top_k) if ('knn' in args.method) or ('kmeans' in args.method) else method_concat
                    method_concat = method_concat + 'pca' + str(args.pca_dims) if args.pca_dims is not None else method_concat
                    if args.norm_emb and args.norm_cfr and args.cfr_no_bias: method_concat += '_normcfr'
                    
                    # Probe training
                    np.random.seed(save_seed)
                    torch.manual_seed(save_seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed(save_seed)
                    # save_seed = save_seed if save_seed!=42 else '' # for backward compat

                    if len(args.dataset_list)==1 and args.ood_test==False:
                        traindata = args.train_file_name
                        if args.train_labels_file_name is not None:
                            if 'se_labels' in args.train_labels_file_name:
                                traindata = args.train_file_name+'_se_labels'
                        probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/{traindata}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                        if 'sampled' in args.test_file_name: probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/{traindata}_{args.test_file_name}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                    elif len(args.dataset_list)>1:
                        multi_name = args.my_multi_name if args.my_multi_name is not None else 'multi2' if len(args.dataset_list)==2 else 'multi'
                        if 'sampledplus' in args.train_name_list[0]: multi_name += 'sampledplus'
                        probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/{multi_name}_/{test_dataset_name}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                    elif args.ood_test:
                        train_dataset_name = args.train_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa')
                        traindata = train_dataset_name
                        if args.train_labels_file_name is not None:
                            if 'se_labels' in args.train_labels_file_name:
                                traindata = train_dataset_name+'_se_labels'
                        probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/ood_{traindata}/_{test_dataset_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                        if 'sampled' in args.test_file_name: probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/ood_{traindata}/_{args.test_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                        if 'dola' in args.test_file_name: probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/ood_{traindata}/_{args.test_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                    plot_name_concat = 'b' if args.use_best_val_t else ''
                    plot_name_concat += 'a' if args.best_using_auc else ''
                    plot_name_concat += 'l' if args.best_as_last else ''
                    probes_file_name += plot_name_concat
                    print(probes_file_name)
                    # Create dirs if does not exist:
                    if not os.path.exists(f'{args.save_path}/probes/models/{probes_file_name}'):
                        os.makedirs(f'{args.save_path}/probes/models/{probes_file_name}', exist_ok=True)
                    if not os.path.exists(f'{args.save_path}/probes/{probes_file_name}'):
                        os.makedirs(f'{args.save_path}/probes/{probes_file_name}', exist_ok=True)

                    # Individual probes
                    all_supcon_train_loss, all_supcon_val_loss = {}, {}
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
                        cur_probe_train_idxs = train_idxs
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
                                if labels_sample_dist.count(0)==1 or labels_sample_dist.count(3)==1: labels_sample_dist[labels_sample_dist.index(3)] = 0
                                if labels_sample_dist.count(1)==1 or labels_sample_dist.count(2)==1: labels_sample_dist[labels_sample_dist.index(2)] = 1
                                if labels_sample_dist.count(4)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                                if labels_sample_dist.count(0)==1: labels_sample_dist[labels_sample_dist.index(4)] = 0
                                if labels_sample_dist.count(1)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                                train_prompt_idxs, val_prompt_idxs, _, _ = train_test_split(np.arange(num_prompts), labels_sample_dist, stratify=labels_sample_dist, test_size=0.2)
                                # train_set_idxs = np.concatenate([np.arange(k*num_samples,(k*num_samples)+num_samples,1) for k in train_prompt_idxs], axis=0)
                                train_set_idxs = np.concatenate([np.arange(ds_prompt_start_idx+(k*num_samples),ds_prompt_start_idx+(k*num_samples)+num_samples,1) for k in train_prompt_idxs], axis=0)
                                # val_set_idxs = np.concatenate([np.arange(k*num_samples,(k*num_samples)+num_samples,1) for k in val_prompt_idxs], axis=0)
                                # val_set_idxs = np.array([k*num_samples for k in val_prompt_idxs])
                                val_set_idxs = np.concatenate([np.arange(ds_prompt_start_idx+(k*num_samples),ds_prompt_start_idx+(k*num_samples)+num_samples,1) for k in val_prompt_idxs], axis=0) if args.use_val_aug else np.array([ds_prompt_start_idx+(k*num_samples) for k in val_prompt_idxs])
                                # assert len(train_set_idxs) + len(val_set_idxs) == args.len_dataset
                                print(min(train_prompt_idxs),max(train_prompt_idxs))
                                print(min(val_prompt_idxs),max(val_prompt_idxs))
                                print(min(train_set_idxs),max(train_set_idxs))
                                print(min(val_set_idxs),max(val_set_idxs))
                                print(len(labels))
                                print('Hallu in val:',sum([labels[i] for i in val_set_idxs]),'Hallu in train:',sum([labels[i] for i in train_set_idxs]))
                                ds_prompt_start_idx += num_samples*num_prompts
                        else:
                            # train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
                            # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
                            train_set_idxs, val_set_idxs, _, _ = train_test_split(train_idxs, labels, stratify=labels,test_size=0.2)

                        y_train_supcon = np.stack([labels[i] for i in train_set_idxs], axis = 0)
                        y_train = np.stack([[labels[i]] for i in train_set_idxs], axis = 0)
                        y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
                        if args.test_file_name is not None: y_test = np.stack([[labels[i]] for i in test_idxs], axis = 0) if args.num_folds>1 else np.stack([test_labels[i] for i in test_idxs], axis = 0)
                        
                        all_supcon_train_loss[i] = []
                        all_train_loss[i], all_val_loss[i], all_val_auc[i] = [], [], []
                        all_val_accs[i], all_val_f1s[i] = [], []
                        all_test_accs[i], all_test_f1s[i] = [], []
                        all_val_preds[i], all_test_preds[i] = [], []
                        all_y_true_val[i], all_y_true_test[i] = [], []
                        all_val_logits[i], all_test_logits[i] = [], []
                        all_val_sim[i], all_test_sim[i] = [], []
                        model_wise_mc_sample_idxs, probes_saved = [], []
                        num_layers = 18 if 'gemma_2B' in args.model_name else 28 if 'gemma_7B' in args.model_name else 33 if '7B' in args.model_name and args.using_act=='layer' else 33 if '8B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 32 if '8B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0 #raise ValueError("Unknown model size.")
                        loop_layers = range(num_layers-1,-1,-1) if 'reverse' in args.method else range(num_layers)
                        loop_layers = [num_layers-1] if args.last_only else loop_layers
                        if args.using_act=='layer_att_res': loop_layers = range(my_train_acts.shape[1])
                        model_idx = -1
                        for layer in tqdm(loop_layers):
                            loop_heads = range(num_heads) if args.using_act == 'ah' else [0]
                            for head in loop_heads:
                                model_idx += 1
                                if args.skip_to_head is not None:
                                    if model_idx!=args.skip_to_head:
                                        continue
                                if args.excl_ce:
                                    cur_probe_train_idxs = np.array([idx for idx in cur_probe_train_idxs if not any(idx in mc_idxs for mc_idxs in model_wise_mc_sample_idxs)]) # Exclude top-k samples of previous probe from current train pool
                                    cur_probe_train_set_idxs = np.random.choice(cur_probe_train_idxs, size=int(len(cur_probe_train_idxs)*(1-0.2)), replace=False) # Split current train pool to train-set and val-set
                                    val_set_idxs = np.array([x for x in cur_probe_train_idxs if x not in cur_probe_train_set_idxs])
                                else:
                                    cur_probe_train_set_idxs = train_set_idxs
                                    val_set_idxs = val_set_idxs
                                cur_probe_y_train = np.stack([[labels[i]] for i in cur_probe_train_set_idxs], axis = 0)
                                y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
                                train_target = np.stack([labels[j] for j in cur_probe_train_set_idxs], axis = 0)
                                class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                                weight = 1. / class_sample_count
                                samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                                ds_train = Dataset.from_dict({"inputs_idxs": cur_probe_train_set_idxs, "labels": cur_probe_y_train}).with_format("torch")
                                ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler) if args.no_batch_sampling==False else DataLoader(ds_train, batch_size=args.bs)
                                ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                                ds_val = DataLoader(ds_val, batch_size=args.bs)
                                if args.test_file_name is not None: 
                                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                                    ds_test = DataLoader(ds_test, batch_size=args.bs)

                                act_dims = {'layer_att_res':2048,'layer':2048,'mlp':None,'mlp_l1':None,'ah':256} if 'gemma_2B' in args.model_name else {'layer_att_res':3072,'layer':3072,'mlp':None,'mlp_l1':None,'ah':128} if 'gemma_7B' in args.model_name else {'layer_att_res':4096,'layer':4096,'mlp':4096,'mlp_l1':11008,'ah':128}
                                bias = False if 'specialised' in args.method or 'orthogonal' in args.method or args.no_bias else True
                                supcon = True if 'supcon' in args.method else False
                                nlinear_model = LogisticRegression_Torch(n_inputs=act_dims[args.using_act], n_outputs=1, bias=bias, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device) if 'individual_linear' in args.method else My_SupCon_NonLinear_Classifier4(input_size=act_dims[args.using_act], output_size=1, bias=bias, use_dropout=args.use_dropout, supcon=supcon, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device) if 'non_linear_4' in args.method else None #My_SupCon_NonLinear_Classifier(input_size=act_dims[args.using_act], output_size=1, bias=bias, use_dropout=args.use_dropout, supcon=supcon).to(device)
                                if 'individual_att_pool' in args.method: nlinear_model = Att_Pool_Layer(llm_dim=act_dims[args.using_act], n_outputs=1)
                                # nlinear_model = My_SupCon_NonLinear_Classifier_wProj(input_size=act_dims[args.using_act], output_size=1, bias=bias, use_dropout=args.use_dropout).to(device)
                                print('\n\nModel Size')
                                wgts = 0
                                for p in nlinear_model.parameters():
                                    sp = torch.squeeze(p)
                                    print(sp.shape)
                                    num_params = 1
                                    for i in range(len(sp.shape)):
                                        num_params *= sp.shape[i]
                                    wgts += num_params
                                print('\n\n#:',wgts)
                                # sys.exit()                                
                                final_layer_name, projection_layer_name = 'linear' if 'individual_linear' in args.method else 'classifier', 'projection'
                                if args.retrain_full_model_path is not None:
                                    retrain_full_model_path = f'{args.save_path}/probes/models/{args.retrain_full_model_path}_model{i}'
                                    retrain_model_state_dict = torch.load(retrain_full_model_path).state_dict()
                                    with torch.no_grad():
                                        for n,param in nlinear_model.named_parameters():
                                            param.copy_(retrain_model_state_dict[n])
                                wgt_0 = np.sum(cur_probe_y_train)/len(cur_probe_y_train)
                                criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
                                criterion_supcon = SupConLoss(temperature=args.supcon_temp) if 'supconv2' in args.method else NTXentLoss()

                                if args.norm_input:
                                    transform_mean, transform_std = torch.mean(torch.stack([my_train_acts[k][layer] for k in train_set_idxs]), dim=-2), torch.std(torch.stack([my_train_acts[k][layer] for k in train_set_idxs]), dim=-2)
                                    my_train_acts[:,layer,:] = (my_train_acts[:,layer,:]-transform_mean)/transform_std
                                    my_test_acts[:,layer,:] = (my_test_acts[:,layer,:]-transform_mean)/transform_std

                                if args.skip_train==False:
                                    # Sup-Con training
                                    if 'supcon' in args.method:
                                        pass
                                        # print('Sup-Con training...')
                                        # train_loss = []
                                        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                                        # named_params = [] # list(nlinear_model.named_parameters())
                                        # print([n for n,_ in nlinear_model.named_parameters()])
                                        # for n,param in nlinear_model.named_parameters():
                                        #     if final_layer_name in n: # Do not train final layer params
                                        #         param.requires_grad = False
                                        #     else:
                                        #         named_params.append((n,param))
                                        #         param.requires_grad = True
                                        # optimizer_grouped_parameters = [
                                        #     {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.supcon_lr},
                                        #     {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.supcon_lr}
                                        # ]
                                        # # print([n for n,p in named_params])
                                        # ds_train_sc = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train_supcon}).with_format("torch")
                                        # ds_train_sc = DataLoader(ds_train_sc, batch_size=args.supcon_bs, sampler=sampler)
                                        # optimizer = torch.optim.Adam(optimizer_grouped_parameters)
                                        # # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                                        # for epoch in range(args.supcon_epochs):
                                        #     epoch_train_loss = 0
                                        #     nlinear_model.train()
                                        #     for step,batch in enumerate(ds_train_sc):
                                        #         optimizer.zero_grad()
                                        #         activations = []
                                        #         for idx in batch['inputs_idxs']:
                                        #             if args.load_act==False:
                                        #                 if args.fast_mode:
                                        #                     act = my_train_acts[idx][layer]
                                        #                 else:
                                        #                     act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                        #                     file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                        #                     file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                        #                     act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                        #             else:
                                        #                 act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                        #             activations.append(act)
                                        #         inputs = torch.stack(activations,axis=0) if args.token in single_token_types else torch.cat(activations,dim=0)
                                        #         if args.token in single_token_types:
                                        #             targets = batch['labels']
                                        #         elif args.token=='all':
                                        #             targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                                        #         if args.token=='tagged_tokens':
                                        #             targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                                        #         emb = nlinear_model.relu1(nlinear_model.linear1(nlinear_model.dropout(inputs))) if args.use_dropout else nlinear_model.relu1(nlinear_model.linear1(inputs))
                                        #         norm_emb = F.normalize(emb, p=2, dim=-1)
                                        #         emb_projection = nlinear_model.projection(norm_emb)
                                        #         emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                                        #         logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                                        #         loss = criterion_supcon(logits, targets.to(device))
                                        #         epoch_train_loss += loss.item()
                                        #         loss.backward()
                                        #         optimizer.step()
                                        #     # scheduler.step()
                                        #     print(epoch_train_loss)
                                        #     train_loss.append(epoch_train_loss)
                                        # all_supcon_train_loss[i].append(np.array(train_loss))
                                    
                                    # Final layer classifier training
                                    print('Final layer classifier training...')
                                    supcon_train_loss, train_loss, val_loss, val_auc = [], [], [], []
                                    best_val_loss, best_spl_loss, best_val_auc = torch.inf, torch.inf, 0
                                    best_model_state = deepcopy(nlinear_model.state_dict())
                                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                                    named_params = list(nlinear_model.named_parameters())
                                    # named_params = [] # list(nlinear_model.named_parameters())
                                    # for n,param in nlinear_model.named_parameters():
                                    #     if final_layer_name in n: # Always train final layer params
                                    #         named_params.append((n,param))
                                    #         param.requires_grad = True
                                    #         if 'specialised' in args.method: param.register_hook(lambda grad: torch.clamp(grad, -1, 1)) # Clip grads
                                    #     else:
                                    #         if 'supcon' in args.method: # Do not train non-final layer params when using supcon
                                    #             param.requires_grad = False
                                    #             # named_params.append((n,param)) # Debug by training all params
                                    #             # param.requires_grad = True
                                    #         else: # Train all params when not using supcon (Note: projection layer is detached from loss so does not matter)
                                    #             named_params.append((n,param))
                                    #             param.requires_grad = True
                                    #             if 'specialised' in args.method: param.register_hook(lambda grad: torch.clamp(grad, -1, 1)) # Clip grads
                                    optimizer_grouped_parameters = [
                                        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.lr},
                                        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
                                    ]
                                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters) # torch.optim.Adam(optimizer_grouped_parameters)
                                    steps_per_epoch = int(len(train_set_idxs)/args.bs)+1  # number of steps in an epoch
                                    warmup_period = steps_per_epoch * 5
                                    T_max = (steps_per_epoch*args.epochs) - warmup_period # args.epochs-warmup_period
                                    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_period)
                                    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
                                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1) if args.scheduler=='static' else torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_period])
                                    for epoch in range(args.epochs):
                                        epoch_supcon_loss, epoch_train_loss, epoch_spl_loss = 0, 0, 0
                                        nlinear_model.train()
                                        for step,batch in enumerate(ds_train):
                                            optimizer.zero_grad()
                                            activations = []
                                            for idx in batch['inputs_idxs']:
                                                if args.load_act==False:
                                                    if args.fast_mode:
                                                        act = my_train_acts[idx][layer].to(device)
                                                    else:
                                                        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                                        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                                else:
                                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                                activations.append(act)
                                            inputs = torch.stack(activations,axis=0) if args.token in single_token_types else torch.cat(activations,dim=0)
                                            if args.token in single_token_types:
                                                targets = batch['labels']
                                            elif args.token=='all':
                                                # print(step,prompt_tokens[idx],len(prompt_tokens[idx]))
                                                targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                                            if args.token=='tagged_tokens':
                                                # targets = torch.cat([torch.Tensor([y_label for j in range(num_tagged_tokens(tagged_token_idxs[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                                                targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                                            # if 'individual_linear_orthogonal' in args.method or 'individual_linear_specialised' in args.method or ('individual_linear' in args.method and args.no_bias) or args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-1).unsqueeze(-1))/torch.std(inputs, dim=-1).unsqueeze(-1) # mean normalise
                                            # print(torch.mean(inputs, dim=-1).sum())
                                            if 'supcon' in args.method:
                                                # SupCon backward
                                                emb = nlinear_model.forward_upto_classifier(inputs)
                                                norm_emb = F.normalize(emb, p=2, dim=-1)
                                                emb_projection = nlinear_model.projection(norm_emb)
                                                emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                                                if 'supconv2' in args.method:
                                                    supcon_loss = criterion_supcon(emb_projection[:,None,:],torch.squeeze(targets).to(device))
                                                else:
                                                    logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                                                    supcon_loss = criterion_supcon(logits, torch.squeeze(targets).to(device))
                                                epoch_supcon_loss += supcon_loss.item()
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
                                                outputs = nlinear_model.linear(inputs) if 'individual_linear' in args.method else nlinear_model(inputs)
                                                loss = criterion(outputs, targets.to(device).float())
                                                loss.backward()
                                            epoch_train_loss += loss.item()
                                            if 'specialised' in args.method and len(model_wise_mc_sample_idxs)>0:
                                                mean_vectors = []
                                                for idxs in model_wise_mc_sample_idxs: # for each previous model
                                                    if len(idxs)>0:
                                                        acts = []
                                                        for idx in idxs: # compute mean vector of all chosen samples in current layer
                                                            if args.fast_mode:
                                                                act = my_train_acts[idx][layer]
                                                            else:
                                                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                                            acts.append(act)
                                                        acts = torch.stack(acts,axis=0)
                                                        if 'non_linear_2' in args.method:
                                                            acts = nlinear_model.relu1(nlinear_model.linear1(acts)) # pass through model up to classifier
                                                        elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                                                            pass
                                                        mean_vectors.append(torch.mean(acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1), dim=0)) # unit normalise and get mean vector
                                                mean_vectors = torch.stack(mean_vectors,axis=0)
                                                # Note: with bce, there is only one probe, i.e only one weight vector
                                                cur_wgts = nlinear_model.classifier.weight if 'non_linear' in args.method else nlinear_model.linear.weight
                                                cur_norm_weights_0 = cur_wgts / cur_wgts.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                                                spl_loss = torch.mean(
                                                                    torch.maximum(torch.zeros(mean_vectors.shape[0]).to(device)
                                                                                ,torch.sum(mean_vectors.data * cur_norm_weights_0, dim=-1)
                                                                                )
                                                                    ) # compute sim and take only positive values
                                                loss = loss + args.spl_wgt*spl_loss
                                                epoch_spl_loss += spl_loss.item()
                                            if 'orthogonal' in args.method and len(probes_saved)>0:
                                                spl_loss = 0
                                                # Note: with bce, there is only one probe, i.e only one weight vector
                                                cur_wgts = nlinear_model.classifier.weight if 'non_linear' in args.method else nlinear_model.linear.weight
                                                cur_norm_weights_0 = cur_wgts / cur_wgts.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                                                for prev_probe_path in probes_saved:
                                                    prev_probe = torch.load(prev_probe_path)
                                                    prev_wgts = prev_probe.classifier.weight if 'non_linear' in args.method else prev_probe.linear.weight
                                                    prev_norm_weights_0 = prev_wgts / prev_wgts.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                                                    # spl_loss += torch.abs(torch.sum(prev_norm_weights_0.data * cur_norm_weights_0, dim=-1))
                                                    spl_loss += torch.maximum(torch.zeros(prev_norm_weights_0.shape[0]).to(device)
                                                                            ,torch.sum(prev_norm_weights_0.data * cur_norm_weights_0, dim=-1)
                                                                            )
                                                loss = loss + args.spl_wgt*spl_loss
                                                epoch_spl_loss += spl_loss.item()
                                            # train_loss.append(loss.item())
                                            # for n,p in nlinear_model.named_parameters():
                                            #     if layer==3 and p.grad is not None: print(step,n,torch.min(p.grad),torch.max(p.grad))
                                            optimizer.step()
                                            scheduler.step()
                                        if 'supcon' in args.method: epoch_supcon_loss = epoch_supcon_loss/(step+1)
                                        epoch_train_loss = epoch_train_loss/(step+1)

                                        
                                        # After each epoch, print mean similarity to top-k samples from first epoch
                                        # if 'specialised' in args.method:
                                        if bias==False:
                                            # Note: with bce, there is only one probe, i.e only one weight vector
                                            cur_wgts = nlinear_model.classifier.weight if 'non_linear' in args.method else nlinear_model.linear.weight
                                            cur_norm_weights_0 = cur_wgts / cur_wgts.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                                            if epoch==0:
                                                acts = []
                                                for idx in cur_probe_train_set_idxs:
                                                    if labels[idx]==hallu_cls:
                                                        if args.fast_mode:
                                                            act = my_train_acts[idx][layer]
                                                        else:
                                                            file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                            file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                            act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                                        acts.append(act)
                                                acts = torch.stack(acts,axis=0)
                                                if 'non_linear_2' in args.method:
                                                    acts = nlinear_model.relu1(nlinear_model.linear1(acts)) # pass through model up to classifier
                                                elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                                                    pass
                                                    # acts = nlinear_model.relu1(nlinear_model.linear1(inputs))
                                                norm_acts = acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                                sim = torch.sum(norm_acts * cur_norm_weights_0, dim=-1)
                                                knn_k = sim.shape[0] if sim.shape[0]<args.spl_knn else args.spl_knn
                                                top_sim_acts = norm_acts[torch.topk(sim,knn_k)[1]]
                                                print(top_sim_acts.shape)
                                            print(torch.mean(torch.sum(top_sim_acts * cur_norm_weights_0, dim=-1)))

                                        # Get val loss
                                        nlinear_model.eval()
                                        epoch_val_loss = 0
                                        val_preds, val_true = [], []
                                        for step,batch in enumerate(ds_val):
                                            optimizer.zero_grad()
                                            activations = []
                                            for idx in batch['inputs_idxs']:
                                                if args.load_act==False:
                                                    if args.fast_mode:
                                                        act = my_train_acts[idx][layer].to(device)
                                                    else:
                                                        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                                        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                                else:
                                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                                activations.append(act)
                                            inputs = torch.stack(activations,axis=0) if args.token in single_token_types else torch.cat(activations,dim=0)
                                            if args.token in single_token_types:
                                                targets = batch['labels']
                                            elif args.token=='all':
                                                # print(step,prompt_tokens[idx],len(prompt_tokens[idx]))
                                                targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                                            if args.token=='tagged_tokens':
                                                # targets = torch.cat([torch.Tensor([y_label for j in range(num_tagged_tokens(tagged_token_idxs[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                                                targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                                            # if 'individual_linear_orthogonal' in args.method or 'individual_linear_specialised' in args.method or ('individual_linear' in args.method and args.no_bias) or args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-1).unsqueeze(-1))/torch.std(inputs, dim=-1).unsqueeze(-1) # mean normalise
                                            if ('knn' in args.method) or ('kmeans' in args.method):
                                                outputs = nlinear_model.forward_upto_classifier(inputs)
                                                epoch_val_loss += 0
                                                if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                    train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                    train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                else:
                                                    train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                    train_labels = None
                                                train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                val_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,train_labels,args.dist_metric,args.top_k) if args.token in single_token_types else None
                                            else:
                                                outputs = nlinear_model(inputs)
                                                epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
                                                val_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in single_token_types else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                                            val_preds += val_preds_batch.tolist()
                                            val_true += batch['labels'].tolist()
                                        epoch_val_loss = epoch_val_loss/(step+1)
                                        # epoch_val_auc = roc_auc_score(val_true, [-v for v in val_preds]) if 'knn' in args.method else roc_auc_score(val_true, val_preds)
                                        epoch_val_auc = roc_auc_score(val_true, val_preds)
                                        supcon_train_loss.append(epoch_supcon_loss)
                                        train_loss.append(epoch_train_loss)
                                        val_loss.append(epoch_val_loss)
                                        val_auc.append(epoch_val_auc)
                                        # print(epoch_spl_loss, epoch_supcon_loss, epoch_train_loss, epoch_val_loss)
                                        # Choose best model
                                        if 'specialised' in args.method:
                                            if epoch_spl_loss < best_spl_loss:
                                                best_spl_loss = epoch_spl_loss
                                                best_model_state = deepcopy(nlinear_model.state_dict())
                                        else:
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
                                            
                                            if args.save_probes:
                                                probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_epoch{epoch}_model{i}_{layer}_{head}'
                                                torch.save(nlinear_model, probe_save_path)
                                        # Early stopping
                                        # patience, min_val_loss_drop, is_not_decreasing = 5, 0.01, 0
                                        # if len(val_loss)>=patience:
                                        #     for epoch_id in range(1,patience,1):
                                        #         val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                                        #         if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                                        #     if is_not_decreasing==patience-1: break
                                    all_supcon_train_loss[i].append(np.array(supcon_train_loss))
                                    all_train_loss[i].append(np.array(train_loss))
                                    all_val_loss[i].append(np.array(val_loss))
                                    all_val_auc[i].append(np.array(val_auc))
                                    nlinear_model.load_state_dict(best_model_state)

                                    # Print similarity of top-k samples at the end of training
                                    # if 'specialised' in args.method:
                                    if bias==False:
                                        hallu_idxs, acts = [], []
                                        for idx in cur_probe_train_idxs: # Identify top-k samples of current train pool
                                            if labels[idx]==hallu_cls:
                                                hallu_idxs.append(idx)
                                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                                acts.append(act)
                                        acts = torch.stack(acts,axis=0)
                                        if 'non_linear_2' in args.method:
                                            acts = nlinear_model.relu1(nlinear_model.linear1(acts)) # pass through model up to classifier
                                        elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                                            pass
                                        norm_acts = acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                        # Note: with bce, there is only one probe, i.e only one weight vector
                                        cur_wgts = nlinear_model.classifier.weight if 'non_linear' in args.method else nlinear_model.linear.weight
                                        cur_norm_weights_0 = cur_wgts / cur_wgts.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                                        sim = torch.sum(norm_acts * cur_norm_weights_0, dim=-1)
                                        knn_k = sim.shape[0] if sim.shape[0]<args.spl_knn else args.spl_knn
                                        top_k = torch.topk(sim,knn_k)[1][torch.topk(sim,knn_k)[0]>0].detach().cpu().numpy() # save indices of top k similar vectors (only pos)
                                        cur_knn_idxs = np.array(hallu_idxs)[top_k]
                                        # model_wise_mc_sample_idxs.append(cur_knn_idxs)
                                        print('Similarity of knn samples at current layer:',sim[top_k])
                                        print('Indices of knn samples at current layer:',cur_knn_idxs)
                                        if sum(torch.topk(sim,knn_k)[0]>0)>0:
                                            top_k_val = torch.min(torch.topk(sim,knn_k)[0][torch.topk(sim,knn_k)[0]>0]).item() # get smallest top-k sim val (only pos)
                                            top_k = (sim>=top_k_val).nonzero(as_tuple=True)[0].detach().cpu().numpy() # save indices of top k similar vectors
                                            cur_knn_idxs = np.array(hallu_idxs)[top_k]
                                            model_wise_mc_sample_idxs.append(cur_knn_idxs)
                                            print('Number of top-k samples:',len(cur_knn_idxs))
                                    
                                    # print(np.array(val_loss))
                                    if args.save_probes:
                                        nlinear_model.load_state_dict(best_model_state)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_model{i}_{layer}_{head}'
                                        torch.save(nlinear_model, probe_save_path)
                                        probes_saved.append(probe_save_path)

                                        nlinear_model.load_state_dict(best_model_state_using_auc)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusingauc_model{i}_{layer}_{head}'
                                        torch.save(nlinear_model, probe_save_path)

                                        nlinear_model.load_state_dict(best_model_state_using_last)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusinglast_model{i}_{layer}_{head}'
                                        torch.save(nlinear_model, probe_save_path)

                                        nlinear_model.load_state_dict(best_model_state_using_loss)
                                        probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_bestusingloss_model{i}_{layer}_{head}'
                                        torch.save(nlinear_model, probe_save_path)
                                    
                                    nlinear_model.load_state_dict(best_model_state)
                                
                                if args.skip_train:
                                    if 'knn' in args.method or 'kmeans' in args.method:
                                        prior_probes_file_name = probes_file_name.replace('knn_','').replace('kmeans_','').replace(args.dist_metric+str(args.top_k)+'_','').replace(args.dist_metric+str(args.top_k)+'pca'+str(args.pca_dims)+'_','')
                                    elif args.ood_test:
                                        traindata = args.train_file_name
                                        if args.train_labels_file_name is not None:
                                            if 'se_labels' in args.train_labels_file_name:
                                                traindata = args.train_file_name+'_se_labels'
                                        prior_probes_file_name = f'NLSC{save_seed}_/{args.model_name}_/{traindata}_/{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                                        if args.use_val_aug: prior_probes_file_name = prior_probes_file_name.replace('_valaug','')
                                        prior_probes_file_name += plot_name_concat
                                    elif 'sampled' in args.test_file_name and len(args.dataset_list)>1 and 'multi2' in multi_name: # city,player -> trivia sampled
                                        prior_probes_file_name = probes_file_name.replace(test_dataset_name,'trivia_qa' if args.multi_probe_dataset_name is None else args.multi_probe_dataset_name)
                                    elif args.use_val_aug: # add aug data to val split
                                        prior_probes_file_name = probes_file_name.replace('_valaug','')
                                    elif 'sampled' in args.test_file_name:
                                        prior_probes_file_name = probes_file_name.replace(args.test_file_name+'_','')
                                    else: # multi
                                        prior_probes_file_name = probes_file_name.replace(test_dataset_name,'trivia_qa' if args.multi_probe_dataset_name is None else args.multi_probe_dataset_name)
                                        if plot_name_concat not in prior_probes_file_name: prior_probes_file_name += plot_name_concat
                                        if args.which_checkpoint not in prior_probes_file_name: prior_probes_file_name += '_' + args.which_checkpoint
                                    try:
                                        prior_save_path = f'{args.save_path}/probes/models/{prior_probes_file_name}_model{i}_{layer}_{head}'
                                        nlinear_model = torch.load(prior_save_path,map_location=device)
                                    except FileNotFoundError:
                                        prior_probes_file_name = prior_probes_file_name.replace("/","") # FOR BACKWARD COMPATIBILITY
                                        prior_save_path = f'{args.save_path}/probes/models/{prior_probes_file_name}_model{i}_{layer}_{head}'
                                        try:
                                            nlinear_model = torch.load(prior_save_path,map_location=device)
                                        except FileNotFoundError:
                                            try:
                                                prior_save_path = prior_save_path.replace("_bestusingauc","")
                                                nlinear_model = torch.load(prior_save_path,map_location=device)
                                            except FileNotFoundError:
                                                prior_save_path = prior_save_path.replace("Falseba_","False_")
                                                nlinear_model = torch.load(prior_save_path,map_location=device)
                                    if args.which_checkpoint not in probes_file_name: probes_file_name += '_' + args.which_checkpoint
                                    probe_save_path = f'{args.save_path}/probes/models/{probes_file_name}_model{i}_{layer}_{head}'
                                    torch.save(nlinear_model, probe_save_path)
                                
                                # Val and Test performance
                                # if args.skip_train==False:
                                if args.skip_train_acts==False:
                                    pred_correct = 0
                                    y_val_pred, y_val_true = [], []
                                    val_preds = []
                                    val_logits = []
                                    val_sim = []
                                    with torch.no_grad():
                                        nlinear_model.eval()
                                        for step,batch in enumerate(ds_val):
                                            activations = []
                                            for idx in batch['inputs_idxs']:
                                                if args.load_act==False:
                                                    if args.fast_mode:
                                                        act = my_train_acts[idx][layer].to(device)
                                                    else:
                                                        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                                        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                                else:
                                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                                activations.append(act)
                                            inputs = torch.stack(activations,axis=0) if args.token in single_token_types else activations
                                            # if 'individual_linear_orthogonal' in args.method or 'individual_linear_specialised' in args.method or ('individual_linear' in args.method and args.no_bias) or args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-1).unsqueeze(-1))/torch.std(inputs, dim=-1).unsqueeze(-1) # mean normalise
                                            if ('knn' in args.method) or ('kmeans' in args.method):
                                                outputs = nlinear_model.forward_upto_classifier(inputs)
                                                # epoch_val_loss += 0
                                                if step==0:
                                                    if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                        train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                        train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                    else:
                                                        train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                        train_labels = np.array([1 for idx in range(len(train_inputs))])
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
                                                # nh_train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs if labels[idx]==0],axis=0) # Take all train non-hallucinations
                                                # outputs = nlinear_model.forward_upto_classifier(nh_train_inputs)
                                                # val_preds_batch = compute_knn_dist(train_outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca) if args.token in single_token_types else None
                                                # print(torch.min(val_preds_batch),torch.quantile(val_preds_batch,0.25),torch.quantile(val_preds_batch,0.5),torch.quantile(val_preds_batch,0.75),torch.quantile(val_preds_batch,0.9),torch.max(val_preds_batch),torch.mean(val_preds_batch))
                                                val_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca) if args.token in single_token_types else None
                                                # print(torch.min(val_preds_batch),torch.quantile(val_preds_batch,0.1),torch.quantile(val_preds_batch,0.25),torch.quantile(val_preds_batch,0.5),torch.quantile(val_preds_batch,0.75),torch.max(val_preds_batch),torch.mean(val_preds_batch))
                                                # sys.exit()
                                                predicted = [1 if v<0.5 else 0 for v in val_preds_batch]
                                            else:
                                                # print('line 1326',inputs.shape)
                                                # predicted = [1 if torch.sigmoid(nlinear_model(inp).data)>0.5 else 0 for inp in inputs] if args.token in single_token_types else torch.stack([1 if torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0]>0.5 else 0 for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                                                val_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in single_token_types else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                                            # y_val_pred += predicted
                                            y_val_true += batch['labels'].tolist()
                                            val_preds.append(val_preds_batch)
                                            if args.token in single_token_types: val_logits.append(nlinear_model(inputs))
                                            if args.token in ['all','tagged_tokens']: val_logits.append(torch.stack([torch.max(nlinear_model(inp).data, dim=0)[0] for inp in inputs]))
                                    all_val_logits[i].append(torch.cat(val_logits))
                                    val_preds = torch.cat(val_preds).cpu().numpy()
                                    all_val_preds[i].append(val_preds)
                                    all_y_true_val[i].append(y_val_true)
                                    # all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
                                    # if layer==num_layers-1:
                                    #     print('Val F1:',f1_score(y_val_true,y_val_pred),f1_score(y_val_true,y_val_pred,pos_label=0))
                                    #     print('Val AUROC:',"%.3f" % roc_auc_score(y_val_true, val_preds))
                                    #     best_val_t = get_best_threshold(y_val_true, val_preds, True if 'knn' in args.method else False)
                                    #     if 'knn' in args.method:
                                    #         y_val_pred_opt = [1 if v<best_val_t else 0 for v in val_preds] if args.use_best_val_t else y_val_pred
                                    #     else:
                                    #         y_val_pred_opt = [1 if v>best_val_t else 0 for v in val_preds] if args.use_best_val_t else y_val_pred
                                    #     log_val_f1 = np.mean([f1_score(y_val_true,y_val_pred_opt),f1_score(y_val_true,y_val_pred_opt,pos_label=0)])
                                    #     log_val_recall = recall_score(y_val_true,y_val_pred_opt)
                                    #     log_val_auc = roc_auc_score(y_val_true, [-v for v in val_preds]) if 'knn' in args.method else roc_auc_score(y_val_true, val_preds)
                                pred_correct = 0
                                y_test_pred, y_test_true = [], []
                                test_preds = []
                                test_logits = []
                                test_sim = []
                                if args.test_file_name is not None: 
                                    with torch.no_grad():
                                        nlinear_model.eval()
                                        use_prompts = tokenized_prompts if args.num_folds>1 else test_tokenized_prompts
                                        use_answer_token_idxes = answer_token_idxes if args.num_folds>1 else test_answer_token_idxes
                                        use_tagged_token_idxs = tagged_token_idxs if args.num_folds>1 else test_tagged_token_idxs
                                        for step,batch in enumerate(ds_test):
                                            activations = []
                                            for idx in batch['inputs_idxs']:
                                                if args.load_act==False:
                                                    if args.fast_mode:
                                                        act = my_test_acts[idx][layer].to(device)
                                                    else:
                                                        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                                        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                                        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                                        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                                else:
                                                    act = get_llama_activations_bau_custom(model, use_prompts[idx], device, args.using_act, layer, args.token, use_answer_token_idxes[idx], use_tagged_token_idxs[idx])
                                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                                activations.append(act)
                                            inputs = torch.stack(activations,axis=0) if args.token in single_token_types else activations
                                            # if 'individual_linear_orthogonal' in args.method or 'individual_linear_specialised' in args.method or ('individual_linear' in args.method and args.no_bias) or args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                                            # if args.norm_input: inputs = (inputs - torch.mean(inputs, dim=-1).unsqueeze(-1))/torch.std(inputs, dim=-1).unsqueeze(-1) # mean normalise
                                            if ('knn' in args.method) or ('kmeans' in args.method):
                                                outputs = nlinear_model.forward_upto_classifier(inputs)
                                                # epoch_val_loss += 0
                                                # if ('maj' in args.dist_metric) or ('wgtd' in args.dist_metric):
                                                #     train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs],axis=0) # Take all train
                                                #     train_labels = np.array([labels[idx] for idx in train_set_idxs])
                                                # else:
                                                #     train_inputs = torch.stack([my_train_acts[idx][layer].to(device) for idx in train_set_idxs if labels[idx]==1],axis=0) # Take all train hallucinations
                                                #     train_labels = None
                                                # train_outputs = nlinear_model.forward_upto_classifier(train_inputs)
                                                test_preds_batch = compute_knn_dist(outputs.data,train_outputs.data,device,train_labels,args.dist_metric,args.top_k,cluster_centers,cluster_centers_labels,pca) if args.token in single_token_types else None
                                                predicted = [1 if v<0.5 else 0 for v in test_preds_batch]
                                            else:
                                                # predicted = [1 if torch.sigmoid(nlinear_model(inp).data)>0.5 else 0 for inp in inputs] if args.token in single_token_types else torch.stack([1 if torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0]>0.5 else 0 for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                                                test_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in single_token_types else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                                            # y_test_pred += predicted
                                            y_test_true += batch['labels'].tolist()
                                            test_preds.append(test_preds_batch)
                                            if args.token in single_token_types: test_logits.append(nlinear_model(inputs))
                                            if args.token in ['all','tagged_tokens']: test_logits.append(torch.stack([torch.max(nlinear_model(inp).data, dim=0)[0] for inp in inputs]))
                                    test_preds = torch.cat(test_preds).cpu().numpy()
                                    all_test_preds[i].append(test_preds)
                                    all_y_true_test[i].append(y_test_true)
                                    # all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
                                    # if layer==num_layers-1:
                                    #     precision, recall, _ = precision_recall_curve(y_test_true, test_preds)
                                    #     print('AuPR:',"%.3f" % auc(recall,precision))
                                    #     print('F1:',f1_score(y_test_true,y_test_pred),f1_score(y_test_true,y_test_pred,pos_label=0))
                                    #     print('Recall:',"%.3f" % recall_score(y_test_true, y_test_pred))
                                    #     print('AuROC:',"%.3f" % roc_auc_score(y_test_true, test_preds))
                                    #     if 'knn' in args.method:
                                    #         y_test_pred_opt = [1 if v<best_val_t else 0 for v in test_preds] if args.use_best_val_t else y_test_pred
                                    #     else:
                                    #         if args.skip_train_acts==False: y_test_pred_opt = [1 if v>best_val_t else 0 for v in test_preds] if args.use_best_val_t else y_test_pred
                                    #     # log_test_f1 = np.mean([f1_score(y_test_true,y_test_pred_opt),f1_score(y_test_true,y_test_pred_opt,pos_label=0)])
                                    #     # log_test_recall = recall_score(y_test_true, y_test_pred_opt)
                                    #     # log_test_auc = roc_auc_score(y_test_true, [-v for v in test_preds]) if 'knn' in args.method else roc_auc_score(y_test_true, test_preds)
                                    all_test_logits[i].append(torch.cat(test_logits))
                                
                        #     break
                        # break
                    
                    # Free up space
                    if 'knn' in args.method or 'kmeans' in args.method:
                        del train_inputs, train_outputs
                        torch.cuda.empty_cache()

                    if args.skip_train_acts==False:
                        np.save(f'{args.save_path}/probes/{probes_file_name}_val_auc.npy', all_val_auc)
                        # all_val_loss = np.stack([np.stack(all_val_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
                        np.save(f'{args.save_path}/probes/{probes_file_name}_val_loss.npy', all_val_loss)
                        # all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
                        np.save(f'{args.save_path}/probes/{probes_file_name}_train_loss.npy', all_train_loss)
                        np.save(f'{args.save_path}/probes/{probes_file_name}_supcon_train_loss.npy', all_supcon_train_loss)
                        # all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_val_pred.npy', all_val_preds)
                        # # all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
                        # np.save(f'{args.save_path}/probes/{probes_file_name}_val_f1.npy', all_val_f1s)
                        # all_y_true_val = np.stack([np.array(all_y_true_val[i]) for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_val_true.npy', all_y_true_val)
                        # all_val_logits = np.stack([torch.stack(all_val_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_val_logits.npy', all_val_logits)
                        # all_val_sim = np.stack([np.stack(all_val_sim[i]) for i in range(args.num_folds)])
                        # np.save(f'{args.save_path}/probes/NLSC{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_sim.npy', all_val_sim)
                        # all_test_sim = np.stack([np.stack(all_test_sim[i]) for i in range(args.num_folds)])
                        # np.save(f'{args.save_path}/probes/NLSC{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_sim.npy', all_test_sim)

                    if args.test_file_name is not None:
                        all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_test_pred.npy', all_test_preds)
                        # all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
                        # np.save(f'{args.save_path}/probes/{probes_file_name}_test_f1.npy', all_test_f1s)
                        all_y_true_test = np.stack([np.array(all_y_true_test[i]) for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_test_true.npy', all_y_true_test)
                        all_test_logits = np.stack([torch.stack(all_test_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
                        np.save(f'{args.save_path}/probes/{probes_file_name}_test_logits.npy', all_test_logits)

                    if args.plot_name is not None:
                        # probes_file_name = f'NLSC{save_seed}_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}{args.norm_input}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}'
                        val_auc = np.load(f'{args.save_path}/probes/{probes_file_name}_val_auc.npy', allow_pickle=True).item()[0]
                        val_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_val_loss.npy', allow_pickle=True).item()[0]
                        train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_train_loss.npy', allow_pickle=True).item()[0]
                        try:
                            supcon_train_loss = np.load(f'{args.save_path}/probes/{probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()[0]
                        except (FileNotFoundError,KeyError):
                            supcon_train_loss = []
                        

                        if len(loop_layers)>1: val_loss = val_loss[-1] # Last layer only
                        if len(loop_layers)>1: train_loss = train_loss[-1] # Last layer only
                        if len(loop_layers)>1: supcon_train_loss = supcon_train_loss[-1] # Last layer only

                        # if len(val_loss)==1:
                        #     val_auc = val_auc[0]
                        #     val_loss = val_loss[0]
                        #     train_loss = train_loss[0]
                        #     if len(supcon_train_loss)>0: supcon_train_loss = supcon_train_loss[0]

                        # if len(val_loss)!=len(train_loss):
                        #     train_loss_by_epoch = []
                        #     batches = int(len(train_loss)/len(val_loss))
                        #     start_at = 0
                        #     for epoch in range(len(val_loss)):
                        #         train_loss_by_epoch.append(sum(train_loss[start_at:(start_at+batches)]))
                        #         start_at += batches
                        #     train_loss = train_loss_by_epoch

                        # print(len(val_auc))
                        # print(len(val_loss))
                        # print(len(train_loss))
                        # if len(supcon_train_loss)>0: print(len(supcon_train_loss))
                        
                        plt.subplot(1, 2, 1)
                        plt.plot(val_loss, label='val_ce_loss')
                        plt.plot(train_loss, label='train_ce_loss')
                        plt.plot(supcon_train_loss, label='train_supcon_loss')
                        plt.legend(loc="upper left")
                        plt.subplot(1, 2, 2)
                        plt.plot(val_auc, label='val_auc')
                        plt.legend(loc="upper left")
                        # plt.savefig(f'{args.save_path}/testfig.png')

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
                        # "with_pe": args.with_pe,
                        # "num_blocks": args.num_blocks,
                        # "wd": args.wd
                        },
                        name=str(save_seed)+'-'+args.plot_name+'-'+str(args.lr)+plot_name_concat
                        )
                        tbl = wandb.Table(columns=['Val AUC', 'Val Recall', 'Val Macro-F1', 'Test AUC', 'Test Recall', 'Test Macro-F1'],
                                    data=[[log_val_auc, log_val_recall, log_val_f1, log_test_auc, log_test_recall, log_test_f1]])
                        wandb.log({#'chart': plt,
                                    'metrics': tbl
                        })
                        wandb.finish()

if __name__ == '__main__':
    main()