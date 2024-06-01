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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from matplotlib import pyplot as plt

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'hl_llama_7B': 'huggyllama/llama-7b',
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

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def num_tagged_tokens(tagged_token_idxs_prompt):
    return sum([b-a+1 for a,b in tagged_token_idxs_prompt])

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
    parser.add_argument('--method',type=str, default='transfomer')
    parser.add_argument('--use_dropout',type=bool, default=False)
    parser.add_argument('--no_bias',type=bool, default=False)
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--epochs',type=int, default=3)
    parser.add_argument('--lr',type=float, default=0.05)
    parser.add_argument('--optimizer',type=str, default='Adam')
    parser.add_argument('--use_class_wgt',type=bool, default=False)
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
        file_path = f'{args.save_path}/responses/{args.train_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        labels = []
        file_path = f'{args.save_path}/responses/{args.train_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json'
        with open(file_path, 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                labels.append(label)
        labels = labels[:args.len_dataset]
        file_path = f'{args.save_path}/responses/{args.test_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer)
        test_labels = []
        file_path = f'{args.save_path}/responses/{args.test_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json'
        with open(file_path, 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                test_labels.append(label)
    
    hallu_cls = 1 if 'hallu_pos' in args.method else 0

    if args.token=='tagged_tokens':
        tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
        test_tagged_token_idxs = get_token_tags(test_prompts,test_prompt_tokens)
    else:
        tagged_token_idxs,test_tagged_token_idxs = [[] for i in range(len(prompts))],[[] for i in range(len(test_prompts))]
    
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
        act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
        my_train_acts, my_test_acts = [], []
        for idx in train_idxs:
            file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
            file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
            act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
            my_train_acts.append(act)
        if args.test_file_name is not None:
            for idx in test_idxs:
                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
                my_test_acts.append(act)

    method_concat = args.method + '_dropout' if args.use_dropout else args.method
    method_concat = args.method + '_no_bias' if args.no_bias else method_concat

    for i in range(args.num_folds):
        print('Training FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
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
        ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler)
        ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
        ds_val = DataLoader(ds_val, batch_size=args.bs)
        if args.test_file_name is not None: 
            ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
            ds_test = DataLoader(ds_test, batch_size=args.bs)

        act_dims = 4096
        bias = False if 'specialised' in args.method or 'orthogonal' in args.method or args.no_bias else True
        nlinear_model = My_Transformer_Layer(n_inputs=act_dims, n_layers=num_layers, n_outputs=1, bias=bias).to(device)
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
        for epoch in range(args.epochs):
            epoch_train_loss, epoch_spl_loss = 0, 0
            nlinear_model.train()
            for step,batch in enumerate(ds_train):
                optimizer.zero_grad()
                activations = []
                for idx in batch['inputs_idxs']:
                    act = my_train_acts[idx]
                    activations.append(act)
                inputs = torch.stack(activations,axis=0)
                targets = batch['labels']
                outputs = nlinear_model(inputs)
                loss = criterion(outputs, targets.to(device).float())
                epoch_train_loss += loss.item()
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            # Get val loss
            nlinear_model.eval()
            epoch_val_loss = 0
            for step,batch in enumerate(ds_val):
                optimizer.zero_grad()
                activations = []
                for idx in batch['inputs_idxs']:
                    act = my_train_acts[idx]
                    activations.append(act)
                inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
                targets = batch['labels']
                outputs = nlinear_model(inputs)
                epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
            val_loss.append(epoch_val_loss)
            print(epoch_spl_loss, epoch_train_loss, epoch_val_loss)
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
                activations = []
                for idx in batch['inputs_idxs']:
                    act = my_train_acts[idx]
                    activations.append(act)
                inputs = torch.stack(activations,axis=0)
                predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:] to add bs dimension
                y_val_pred += predicted
                y_val_true += batch['labels'].tolist()
                val_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                val_preds.append(val_preds_batch)
                val_logits.append(nlinear_model(inputs))
        all_val_preds[i].append(torch.cat(val_preds).cpu().numpy())
        all_y_true_val[i].append(y_val_true)
        all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
        print('Val F1:',f1_score(y_val_true,y_val_pred),f1_score(y_val_true,y_val_pred,pos_label=0))
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
                        act = my_test_acts[idx]
                        activations.append(act)
                    inputs = torch.stack(activations,axis=0)
                    predicted = [1 if torch.sigmoid(nlinear_model(inp[None,:]).data)>0.5 else 0 for inp in inputs] # inp[None,:] to add bs dimension
                    y_test_pred += predicted
                    y_test_true += batch['labels'].tolist()
                    test_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                    test_preds.append(test_preds_batch)
                    test_logits.append(nlinear_model(inputs))
            all_test_preds[i].append(torch.cat(test_preds).cpu().numpy())
            all_y_true_test[i].append(y_test_true)
            all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
            print('Test F1:',f1_score(y_test_true,y_test_pred),f1_score(y_test_true,y_test_pred,pos_label=0))
            all_test_logits[i].append(torch.cat(test_logits))
        all_val_logits[i].append(torch.cat(val_logits))
    

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