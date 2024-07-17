import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import statistics
import pickle
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, tokenized_from_file_v2, get_token_tags
from utils import My_Transformer_Layer
from copy import deepcopy
import llama
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score, classification_report, precision_recall_curve, auc, roc_auc_score
from matplotlib import pyplot as plt

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

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument('--max_tokens',type=int, default=25)
    parser.add_argument('--tokens_first',type=bool, default=False) # Specifies order of tokens and layers when using_act='tagged_tokens'
    parser.add_argument('--no_sep',type=bool, default=False)
    parser.add_argument('--method',type=str, default='transfomer') # (<_hallu_pos>)
    parser.add_argument('--use_dropout',type=bool, default=False)
    parser.add_argument('--no_bias',type=bool, default=False)
    parser.add_argument('--norm_input',type=bool, default=False)
    parser.add_argument('--supcon_temp',type=float, default=0.1)
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_samples',type=int, default=None)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--epochs',type=int, default=3)
    parser.add_argument('--lr',type=float, default=0.05)
    parser.add_argument('--optimizer',type=str, default='Adam')
    parser.add_argument('--use_class_wgt',type=bool, default=False)
    parser.add_argument('--no_batch_sampling',type=bool, default=False)
    parser.add_argument('--acts_per_file',type=int, default=100)
    parser.add_argument('--save_probes',type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--train_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--fast_mode',type=bool, default=False) # use when GPU space is free, dataset is small and using only 1 token per sample
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
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        # if args.load_act==True: # Only load model if we need activations on the fly
        #     model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        # num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name and args.using_act=='mlp' else None #TODO: update for bigger models
        num_heads = 32
    device = "cuda"

    print("Loading prompts and model responses..")
    if args.dataset_name == 'counselling':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts = tokenized_mi(file_path, tokenizer)
    elif args.dataset_name == 'gsm8k' or args.dataset_name == 'strqa' or ('baseline' in args.train_file_name or 'dola' in args.train_file_name):
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        labels = []
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
            if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
            labels.append(label)
        labels = labels[:args.len_dataset]
        if args.test_file_name is None:
            test_prompts, test_labels = [], [] # No test file
        else:
            file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
            test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file_v2(file_path, tokenizer)
            test_labels = []
            with open(file_path, 'r') as read_file:
                data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
                if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
                test_labels.append(label)
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name == 'tqa_gen':
        num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 10 if 'sampled' in args.train_file_name else 1
        file_path = f'{args.save_path}/responses/{args.train_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, num_samples)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        if 'se_labels' in args.train_labels_file_name:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.train_labels_file_name}.npy'
            labels = np.load(file_path)
        else:
            labels = []
            file_path = f'{args.save_path}/responses/{args.train_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    if 'greedy' in args.train_labels_file_name:
                        if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                        if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        labels.append(label)
                    else:
                        for j in range(1,num_samples+1,1):
                            if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                            if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            labels.append(label)
        labels = labels[:args.len_dataset]
        file_path = f'{args.save_path}/responses/{args.test_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer)
        if 'se_labels' in args.test_labels_file_name:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.test_labels_file_name}.npy'
            test_labels = np.load(file_path)
        else:
            test_labels = []
            file_path = f'{args.save_path}/responses/{args.test_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                    if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                    test_labels.append(label)
    
    hallu_cls = 1 if 'hallu_pos' in args.method else 0

    # if args.token=='tagged_tokens':
    #     tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
    #     test_tagged_token_idxs = get_token_tags(test_prompts,test_prompt_tokens)
    # else:
    #     tagged_token_idxs,test_tagged_token_idxs = [[] for i in range(len(prompts))],[[] for i in range(len(test_prompts))]
    
    if args.dataset_name=='strqa':
        args.acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        args.acts_per_file = 20
    else:
        args.acts_per_file = 100


    # Probe training
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    # Individual probes
    all_supcon_train_loss, all_supcon_val_loss = {}, {}
    all_train_loss, all_val_loss = {}, {}
    all_val_accs, all_val_f1s = {}, {}
    all_test_accs, all_test_f1s = {}, {}
    all_val_preds, all_test_preds = {}, {}
    all_y_true_val, all_y_true_test = {}, {}
    all_val_logits, all_test_logits = {}, {}
    all_val_sim, all_test_sim = {}, {}
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
    
    if args.fast_mode:
        print("Loading acts...")
        my_train_acts, my_test_acts = [], []
        for idx in train_idxs:
            file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
            file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
            if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                act = combine_acts(idx,args.train_file_name,args)
                if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                if args.no_sep==False:
                    sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                    act = torch.cat((act,sep_token), dim=1)
                act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
            else:
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
            my_train_acts.append(act)
        # if args.token=='tagged_tokens': my_train_acts = torch.nn.utils.rnn.pad_sequence(my_train_acts, batch_first=True)
        
        if args.test_file_name is not None:
            for idx in test_idxs:
                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                    # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                    act = combine_acts(idx,args.test_file_name,args)
                    if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                    if args.no_sep==False:
                        sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                        act = torch.cat((act,sep_token), dim=1)
                    act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
                else:
                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                my_test_acts.append(act)
            # if args.token=='tagged_tokens': my_test_acts = torch.nn.utils.rnn.pad_sequence(my_test_acts, batch_first=True)

    method_concat = args.method + '_dropout' if args.use_dropout else args.method
    method_concat = args.method + '_no_bias' if args.no_bias else method_concat

    for i in range(args.num_folds):
        print('Training FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
        if 'sampled' in args.train_file_name:
            num_samples = args.num_samples if args.num_samples is not None else 10
            num_prompts = len(train_idxs)/num_samples
            train_set_idxs = train_idxs[:int(num_prompts*(1-0.2))*num_samples]
            val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        else:
            train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
            val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        y_train_supcon = np.stack([labels[i] for i in train_set_idxs], axis = 0)
        y_train = np.stack([[labels[i]] for i in train_set_idxs], axis = 0)
        y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
        if args.test_file_name is not None: y_test = np.stack([[labels[i]] for i in test_idxs], axis = 0) if args.num_folds>1 else np.stack([test_labels[i] for i in test_idxs], axis = 0)
        
        all_train_loss[i], all_val_loss[i] = [], []
        all_val_accs[i], all_val_f1s[i] = [], []
        all_test_accs[i], all_test_f1s[i] = [], []
        all_val_preds[i], all_test_preds[i] = [], []
        all_y_true_val[i], all_y_true_test[i] = [], []
        all_val_logits[i], all_test_logits[i] = [], []
        all_val_sim[i], all_test_sim[i] = [], []
        model_wise_mc_sample_idxs, probes_saved = [], []
        num_layers = 33 if '7B' in args.model_name and args.using_act=='layer' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0 #raise ValueError("Unknown model size.")
        
        cur_probe_train_set_idxs = train_set_idxs
        cur_probe_y_train = np.stack([[labels[i]] for i in cur_probe_train_set_idxs], axis = 0)
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

        act_dims = 4096
        bias = False if 'specialised' in args.method or 'orthogonal' in args.method or args.no_bias else True
        n_blocks = 2 if 'transformer2' in args.method else 1
        nlinear_model = My_Transformer_Layer(n_inputs=act_dims, n_layers=num_layers, n_outputs=1, bias=bias, n_blocks=n_blocks).to(device)
        wgt_0 = np.sum(cur_probe_y_train)/len(cur_probe_y_train)
        criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
        criterion_supcon = NTXentLoss()
        
        # Training
        train_loss, val_loss = [], []
        best_val_loss = torch.inf
        best_model_state = deepcopy(nlinear_model.state_dict())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        named_params = list(nlinear_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.lr},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        # optimizer = torch.optim.Adam(nlinear_model.parameters())
        for epoch in tqdm(range(args.epochs)):
            num_samples_used, num_val_samples_used, epoch_train_loss, epoch_supcon_loss = 0, 0, 0, 0
            nlinear_model.train()
            for step,batch in enumerate(ds_train):
                optimizer.zero_grad()
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
                        act = my_train_acts[idx]
                    activations.append(act)
                if len(activations)==0: continue
                num_samples_used += len(batch_target_idxs)
                if 'tagged_tokens' in args.token:
                    inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                else:
                    inputs = torch.stack(activations,axis=0)
                if args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                targets = batch['labels'][np.array(batch_target_idxs)] if 'tagged_tokens' in args.token else batch['labels']
                if 'supcon' in args.method:
                    # SupCon backward
                    emb = nlinear_model.forward_upto_classifier(inputs)
                    norm_emb = F.normalize(emb, p=2, dim=-1)
                    emb_projection = nlinear_model.projection(norm_emb)
                    emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                    logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                    supcon_loss = criterion_supcon(logits, targets.to(device))
                    epoch_supcon_loss += loss.item()
                    supcon_loss.backward()
                    # CE backward
                    emb = nlinear_model.forward_upto_classifier(inputs).detach()
                    outputs = nlinear_model.classifier(emb)
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
                epoch_train_loss += loss.item()
                train_loss.append(loss.item())

            # Get val loss
            nlinear_model.eval()
            epoch_val_loss = 0
            for step,batch in enumerate(ds_val):
                optimizer.zero_grad()
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
                        act = my_train_acts[idx]
                    activations.append(act)
                if len(activations)==0: continue
                num_val_samples_used += len(batch_target_idxs)
                if 'tagged_tokens' in args.token:
                    inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                else:
                    inputs = torch.stack(activations,axis=0)
                if args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                targets = batch['labels'][np.array(batch_target_idxs)] if 'tagged_tokens' in args.token else batch['labels']
                outputs = nlinear_model(inputs)
                epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
            val_loss.append(epoch_val_loss)
            print('Loss:', epoch_supcon_loss, epoch_train_loss, epoch_val_loss)
            print('Samples:',num_samples_used, num_val_samples_used)
            # Choose best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = deepcopy(nlinear_model.state_dict())
            # Early stopping
            patience, min_val_loss_drop, is_not_decreasing = 5, 0.01, 0
            if len(val_loss)>=patience:
                for epoch_id in range(1,patience,1):
                    val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                    if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                if is_not_decreasing==patience-1: break
        all_train_loss[i].append(np.array(train_loss))
        all_val_loss[i].append(np.array(val_loss))
        nlinear_model.load_state_dict(best_model_state)
        
        # print(np.array(val_loss))
        if args.save_probes:
            probe_save_path = f'{args.save_path}/probes/models/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_model{i}'
            torch.save(nlinear_model, probe_save_path)
            probes_saved.append(probe_save_path)
        
        # Val and Test performance
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
                        act = my_train_acts[idx]
                    activations.append(act)
                if len(activations)==0: continue
                if 'tagged_tokens' in args.token:
                    inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                else:
                    inputs = torch.stack(activations,axis=0)
                if args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:,:] to add bs dimension
                y_val_pred += predicted
                y_val_true += batch['labels'][np.array(batch_target_idxs)].tolist() if 'tagged_tokens' in args.token else batch['labels'].tolist()
                val_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                val_preds.append(val_preds_batch)
                val_logits.append(nlinear_model(inputs))
        val_preds = torch.cat(val_preds).cpu().numpy()
        all_val_preds[i].append(val_preds)
        all_y_true_val[i].append(y_val_true)
        all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
        all_val_logits[i].append(torch.cat(val_logits))
        print('Val F1:',f1_score(y_val_true,y_val_pred),f1_score(y_val_true,y_val_pred,pos_label=0))
        print('Val AUC:',roc_auc_score(y_val_true, val_preds))
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
                            act = my_test_acts[idx]
                        activations.append(act)
                    if len(activations)==0: continue
                    num_test_samples_used += len(batch_target_idxs)
                    samples_used_idxs += batch['inputs_idxs'][np.array(batch_target_idxs)]
                    if 'tagged_tokens' in args.token:
                        inputs = torch.nn.utils.rnn.pad_sequence(activations, batch_first=True)
                    else:
                        inputs = torch.stack(activations,axis=0)
                    if args.norm_input: inputs = inputs / inputs.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
                    predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:,:] to add bs dimension
                    y_test_pred += predicted
                    y_test_true += batch['labels'][np.array(batch_target_idxs)].tolist() if 'tagged_tokens' in args.token else batch['labels'].tolist()
                    test_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                    test_preds.append(test_preds_batch)
                    test_logits.append(nlinear_model(inputs))
            test_preds = torch.cat(test_preds).cpu().numpy()
            all_test_preds[i].append(test_preds)
            all_y_true_test[i].append(y_test_true)
            all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
            precision, recall, _ = precision_recall_curve(y_test_true, test_preds)
            print('AuPR for cls1:',auc(recall,precision))
            print('Test F1:',f1_score(y_test_true,y_test_pred),f1_score(y_test_true,y_test_pred,pos_label=0))
            print('Recall for cls1:',recall_score(y_test_true, y_test_pred))
            print('AuROC for cls1:',roc_auc_score(y_test_true, test_preds))
            print('Samples:',num_test_samples_used)
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
    

    # all_val_loss = np.stack([np.stack(all_val_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_loss.npy', all_val_loss)
    # all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_train_loss.npy', all_train_loss)
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_supcon_train_loss.npy', all_supcon_train_loss)
    all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_pred.npy', all_val_preds)
    all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_f1.npy', all_val_f1s)
    all_y_true_val = np.stack([np.array(all_y_true_val[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_true.npy', all_y_true_val)
    all_val_logits = np.stack([torch.stack(all_val_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_logits.npy', all_val_logits)
    # all_val_sim = np.stack([np.stack(all_val_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_sim.npy', all_val_sim)
    # all_test_sim = np.stack([np.stack(all_test_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_sim.npy', all_test_sim)
    np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_samples_used.npy', samples_used_idxs)

    if args.test_file_name is not None:
        all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_pred.npy', all_test_preds)
        all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_f1.npy', all_test_f1s)
        all_y_true_test = np.stack([np.array(all_y_true_test[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_true.npy', all_y_true_test)
        all_test_logits = np.stack([torch.stack(all_test_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/T_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_logits.npy', all_test_logits)

if __name__ == '__main__':
    main()