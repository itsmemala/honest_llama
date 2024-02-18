import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import sys
sys.path.append('../')
from utils import get_separated_activations, train_mlp_probes, train_mlp_single_probe
import llama

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf', 
    'honest_llama_7B': 'results_dump/llama_7B_seed_42_top_48_heads_alpha_15', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'honest_alpaca_7B': 'results_dump/alpaca_7B_seed_42_top_48_heads_alpha_15', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'honest_vicuna_7B': 'results_dump/vicuna_7B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'honest_llama2_chat_7B': 'results_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'honest_llama2_chat_13B': 'results_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'honest_llama2_chat_70B': 'results_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15',
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--type_probes', type=str, default='ind')
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataset
    if args.dataset_name == "tqa_mc2":
        # dataset = load_dataset("truthful_qa", "multiple_choice", streaming= True)['validation']
        len_dataset = 817
    elif args.dataset_name=='tqa_gen':
        # dataset = load_dataset("truthful_qa", "generation", streaming= True)['validation']
        len_dataset = 817
    elif args.dataset_name=='nq':
        # dataset = load_dataset("OamPatel/iti_nq_open_val", streaming= True)['validation']
        len_dataset = 3610
    elif args.dataset_name=='counselling':
        len_dataset = 500

    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len_dataset), args.num_fold)

    # create model
    # model_name = HF_NAMES["honest_" + args.model_name if args.use_honest else args.model_name]
    # MODEL = model_name if not args.model_dir else args.model_dir
    # tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    # model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    
    # define number of layers and heads
    num_layers = 60 # model.config.num_hidden_layers
    num_heads = 52 # model.config.num_attention_heads
    num_dim = 6656

    # load activations
    try:
        mlp_wise_activations = np.load(f"{args.save_path}/features/{args.model_name}_{args.dataset_name}_mlp_wise.npy")
        labels = np.load(f"{args.save_path}/features/{args.model_name}_{args.dataset_name}_labels.npy")
    except FileNotFoundError:
        if 'tqa' in args.dataset_name:
            file_ends = [1000,3000,4000,5000,6000]
        elif 'nq' in args.dataset_name:
            file_ends = [1000,3000,5000,7000,9000,11000]
        elif 'counselling' in args.dataset_name:
            file_ends = [30,60,90,120,150,180] + [i for i in range(200,520,20)]
        mlp_wise_activations = []
        for file_end in file_ends:
            mlp_wise_activations.append(np.load(f"{args.save_path}/features/{args.model_name}_{args.dataset_name}_mlp_wise_{file_end}.npy"))
        mlp_wise_activations = np.concatenate(mlp_wise_activations, axis=0)
        assert mlp_wise_activations.shape[1:] == (num_layers, num_dim)
        labels = np.load(f"{args.save_path}/responses/{args.model_name}_annomi_greedy_responses_{file_end}_parrotting_labels.npy")
        assert len(labels)==len(mlp_wise_activations)

    # separated_mlp_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, mlp_wise_activations, args.dataset_name)

    # run k-fold cross validation
    results = []
    probe_coefs = []
    all_y_val_pred = []
    all_y_val = []
    train_set_idxs_foldwise = []
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        train_set_idxs_foldwise.append([labels[i] for i in train_set_idxs])

        # train probes
        if args.type_probes=='ind':
            probes, curr_fold_results = train_mlp_probes(args.seed, train_set_idxs, val_set_idxs, mlp_wise_activations, labels, num_layers, args.type_probes, sep_act=False)
            cur_probe_coefs = []
            for probe in probes:
                cur_probe_coefs += list(probe.coef_[0])
            probe_coefs.append(cur_probe_coefs)
        elif args.type_probes=='vote_on_ind':
            probes, curr_fold_results, cur_y_val_pred, cur_y_val = train_mlp_probes(args.seed, train_set_idxs, val_set_idxs, mlp_wise_activations, labels, num_layers, args.type_probes, sep_act=False)
            cur_y_val_pred = cur_y_val_pred.swapaxes(0,1)
            all_y_val_pred.append(cur_y_val_pred)
            all_y_val.append(cur_y_val)    
        elif args.type_probes=='lr_on_ind':
            probes, curr_fold_results = train_mlp_probes(args.seed, train_set_idxs, val_set_idxs, mlp_wise_activations, labels, num_layers, args.type_probes, sep_act=False)
        else:
            probe, curr_fold_results, y_val_pred, y_val = train_mlp_single_probe(args.seed, train_set_idxs, val_set_idxs, mlp_wise_activations, labels, num_layers, sep_act=False)
            probe_coefs.append(list(probe.coef_[0]))
            all_y_val_pred.append(y_val_pred)
            all_y_val.append(y_val)

        print(f"FOLD {i}")
        # print(curr_fold_results)

        results.append(curr_fold_results)        
    
    results = np.array(results)
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.dataset_name}_{args.num_fold}_{args.type_probes}_mlp_probe_accs.npy', results)
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.dataset_name}_{args.num_fold}_{args.type_probes}_mlp_probe_true_train.npy',train_set_idxs_foldwise)
    if args.type_probes=='single' or args.type_probes=='ind':
        np.save(f'{args.save_path}/probes/{args.model_name}_{args.dataset_name}_{args.num_fold}_{args.type_probes}_mlp_probe_coef.npy', probe_coefs)
    if args.type_probes=='vote_on_ind' or args.type_probes=='single':
        np.save(f'{args.save_path}/probes/{args.model_name}_{args.dataset_name}_{args.num_fold}_{args.type_probes}_mlp_probe_pred.npy', all_y_val_pred)
        np.save(f'{args.save_path}/probes/{args.model_name}_{args.dataset_name}_{args.num_fold}_{args.type_probes}_mlp_probe_true.npy', all_y_val)
    final = results.mean(axis=0)
    # print('Mean Across Folds:',final)

if __name__ == "__main__":
    main()
