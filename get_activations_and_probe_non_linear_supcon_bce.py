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
from utils import My_SupCon_NonLinear_Classifier, My_SupCon_NonLinear_Classifier4
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
    parser.add_argument('--method',type=str, default='individual_non_linear_3') # individual_non_linear_2, individual_non_linear_3 (<_supcon>, <_specialised>, <_hallu_pos>) 
    parser.add_argument('--supcon_temp',type=float, default=0.1)
    parser.add_argument('--spl_wgt',type=float, default=1)
    parser.add_argument('--spl_knn',type=int, default=5)
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--supcon_bs',type=int, default=128)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--supcon_epochs',type=int, default=10)
    parser.add_argument('--epochs',type=int, default=3)
    parser.add_argument('--supcon_lr',type=float, default=0.05)
    parser.add_argument('--lr',type=float, default=0.05)
    parser.add_argument('--optimizer',type=str, default='Adam')
    parser.add_argument('--use_class_wgt',type=bool, default=False)
    parser.add_argument('--load_act',type=bool, default=False)
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
        if args.load_act==True: # Only load model if we need activations on the fly
            model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        num_layers = 32
        num_heads = 32
    device = "cuda"

    print("Loading prompts and model responses..")
    if args.dataset_name == 'counselling':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts = tokenized_mi(file_path, tokenizer)
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        labels = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json', 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                labels.append(label)
        labels = labels[:args.len_dataset]
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer)
        test_labels = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json', 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                test_labels.append(label)
    elif args.dataset_name == 'gsm8k' or args.dataset_name == 'strqa':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        labels = []
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            label = 1 if data['is_correct'][i]==True else 0
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
                label = 1 if data['is_correct'][i]==True else 0
                test_labels.append(label)
    
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
    
    method_concat = args.method + '_' + str(args.supcon_bs) + '_' + str(args.supcon_epochs) + '_' + str(args.supcon_lr) + '_' + str(args.supcon_temp) if 'supcon' in args.method else args.method
    method_concat = method_concat + '_' + str(args.spl_wgt) + '_' + str(args.spl_knn) if 'specialised' in args.method else method_concat

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
        
        all_supcon_train_loss[i], all_supcon_val_loss[i] = [], []
        all_train_loss[i], all_val_loss[i] = [], []
        all_val_accs[i], all_val_f1s[i] = [], []
        all_test_accs[i], all_test_f1s[i] = [], []
        all_val_preds[i], all_test_preds[i] = [], []
        all_y_true_val[i], all_y_true_test[i] = [], []
        all_val_logits[i], all_test_logits[i] = [], []
        all_val_sim[i], all_test_sim[i] = [], []
        model_wise_mc_sample_idxs = []
        num_layers = 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0 #raise ValueError("Unknown model size.")
        for layer in tqdm(range(num_layers)):
            loop_heads = range(num_heads) if args.using_act == 'ah' else [0]
            for head in loop_heads:
                # if 'individual_non_linear' in args.method:
                train_target = np.stack([labels[j] for j in train_set_idxs], axis = 0)
                class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                weight = 1. / class_sample_count
                samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                ds_train = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train}).with_format("torch")
                ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler)
                ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                ds_val = DataLoader(ds_val, batch_size=args.bs)
                if args.test_file_name is not None: 
                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                    ds_test = DataLoader(ds_test, batch_size=args.bs)

                act_dims = {'layer':4096,'mlp':4096,'mlp_l1':11008,'ah':128}
                nlinear_model = My_SupCon_NonLinear_Classifier4(input_size=act_dims[args.using_act], output_size=1).to(device) if 'non_linear_4' in args.method else My_SupCon_NonLinear_Classifier(input_size=act_dims[args.using_act], output_size=1).to(device)
                final_layer_name, projection_layer_name = 'classifier', 'projection'
                wgt_0 = np.sum(y_train)/len(y_train)
                criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
                criterion_supcon = NTXentLoss()

                # Sup-Con training
                if 'supcon' in args.method:
                    print('Sup-Con training...')
                    train_loss = []
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    named_params = [] # list(nlinear_model.named_parameters())
                    print([n for n,_ in nlinear_model.named_parameters()])
                    for n,param in nlinear_model.named_parameters():
                        if final_layer_name in n: # Do not train final layer params
                            param.requires_grad = False
                        else:
                            named_params.append((n,param))
                            param.requires_grad = True
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.supcon_lr},
                        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.supcon_lr}
                    ]
                    # print([n for n,p in named_params])
                    ds_train_sc = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train_supcon}).with_format("torch")
                    ds_train_sc = DataLoader(ds_train_sc, batch_size=args.supcon_bs, sampler=sampler)
                    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                    for epoch in range(args.supcon_epochs):
                        epoch_train_loss = 0
                        nlinear_model.train()
                        for step,batch in enumerate(ds_train_sc):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
                            if args.token in ['answer_last','prompt_last','maxpool_all']:
                                targets = batch['labels']
                            elif args.token=='all':
                                targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                            if args.token=='tagged_tokens':
                                targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                            emb = nlinear_model.relu1(nlinear_model.linear1(inputs))
                            norm_emb = F.normalize(emb, p=2, dim=-1)
                            emb_projection = nlinear_model.projection(norm_emb)
                            emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                            logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                            loss = criterion_supcon(logits, targets.to(device))
                            epoch_train_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        # scheduler.step()
                        print(epoch_train_loss)
                        train_loss.append(epoch_train_loss)
                    all_supcon_train_loss[i].append(np.array(train_loss))
                
                # Final layer classifier training
                print('Final layer classifier training...')
                train_loss, val_loss = [], []
                best_val_loss = torch.inf
                best_model_state = deepcopy(nlinear_model.state_dict())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                named_params = [] # list(nlinear_model.named_parameters())
                for n,param in nlinear_model.named_parameters():
                    if final_layer_name in n: # Always train final layer params
                        named_params.append((n,param))
                        param.requires_grad = True
                    else:
                        if 'supcon' in args.method: # Do not train non-final layer params when using supcon
                            param.requires_grad = False
                            # named_params.append((n,param)) # Debug by training all params
                            # param.requires_grad = True
                        else: # Train all params when not using supcon (Note: projection layer is detached from loss so does not matter)
                            named_params.append((n,param))
                            param.requires_grad = True
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
                            if args.load_act==False:
                                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                            else:
                                act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                            activations.append(act)
                        inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
                        if args.token in ['answer_last','prompt_last','maxpool_all']:
                            targets = batch['labels']
                        elif args.token=='all':
                            # print(step,prompt_tokens[idx],len(prompt_tokens[idx]))
                            targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                        if args.token=='tagged_tokens':
                            # targets = torch.cat([torch.Tensor([y_label for j in range(num_tagged_tokens(tagged_token_idxs[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                            targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                        outputs = nlinear_model(inputs)
                        loss = criterion(outputs, targets.to(device).float())
                        epoch_train_loss += loss.item()
                        if 'specialised' in args.method and len(model_wise_mc_sample_idxs)>0:
                            if step==0: # Only load once to save time
                                mean_vectors = []
                                for idxs in model_wise_mc_sample_idxs: # for each previous model
                                    acts = []
                                    for idx in idxs: # compute mean vector of all chosen samples in current layer
                                        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                        acts.append(act)
                                    acts = torch.stack(acts,axis=0)
                                    if 'non_linear_2' in args.method:
                                        acts = nlinear_model.relu1(nlinear_model.linear1(inputs)) # pass through model up to classifier
                                    elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                                        pass
                                    mean_vectors.append(torch.mean(acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1), dim=0)) # unit normalise and get mean vector
                                mean_vectors = torch.stack(mean_vectors,axis=0)
                            # Note: with bce, there is only one probe, i.e only one weight vector
                            cur_norm_weights_0 = nlinear_model.classifier.weight / nlinear_model.classifier.weight.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                            spl_loss = torch.mean(
                                                torch.maximum(torch.zeros(mean_vectors.shape[0]).to(device)
                                                            ,torch.sum(mean_vectors * cur_norm_weights_0, dim=-1)
                                                            )
                                                ) # compute sim and take only positive values
                            loss = loss + args.spl_wgt*spl_loss
                            epoch_spl_loss.append(spl_loss.item())
                        train_loss.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    
                    # Specialise
                    if 'specialised' in args.method:
                        # Note: with bce, there is only one probe, i.e only one weight vector
                        cur_norm_weights_0 = nlinear_model.classifier.weight / nlinear_model.classifier.weight.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        if epoch==0:
                            acts = []
                            for idx in train_set_idxs:
                                if labels[idx]==0:
                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                                    acts.append(act)
                            acts = torch.stack(acts,axis=0)
                            if 'non_linear_2' in args.method:
                                acts = nlinear_model.relu1(nlinear_model.linear1(inputs)) # pass through model up to classifier
                            elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                                pass
                                # acts = nlinear_model.relu1(nlinear_model.linear1(inputs))
                            norm_acts = acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                            sim = torch.sum(norm_acts * cur_norm_weights_0, dim=-1)
                            top_sim_acts = norm_acts[torch.topk(sim,args.spl_knn)[1]]
                            print(top_sim_acts.shape)
                        print(torch.mean(torch.sum(top_sim_acts * cur_norm_weights_0, dim=-1)))

                    # Get val loss
                    nlinear_model.eval()
                    epoch_val_loss = 0
                    for step,batch in enumerate(ds_val):
                        optimizer.zero_grad()
                        activations = []
                        for idx in batch['inputs_idxs']:
                            if args.load_act==False:
                                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                            else:
                                act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                            activations.append(act)
                        inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
                        if args.token in ['answer_last','prompt_last','maxpool_all']:
                            targets = batch['labels']
                        elif args.token=='all':
                            # print(step,prompt_tokens[idx],len(prompt_tokens[idx]))
                            targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                        if args.token=='tagged_tokens':
                            # targets = torch.cat([torch.Tensor([y_label for j in range(num_tagged_tokens(tagged_token_idxs[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0).type(torch.LongTensor)
                            targets = torch.cat([torch.Tensor([y_label for j in range(activations[b_idx].shape[0])]) for b_idx,(idx,y_label) in enumerate(zip(batch['inputs_idxs'],batch['labels']))],dim=0).type(torch.LongTensor)
                        outputs = nlinear_model(inputs)
                        epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
                    val_loss.append(epoch_val_loss)
                    print(epoch_spl_loss, epoch_train_loss, epoch_val_loss)
                    # Choose best model
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        best_model_state = deepcopy(nlinear_model.state_dict())
                all_train_loss[i].append(np.array(train_loss))
                all_val_loss[i].append(np.array(val_loss))
                nlinear_model.load_state_dict(best_model_state)

                if 'specialised' in args.method:
                    hallu_idxs, acts = [], []
                    for idx in train_set_idxs:
                        if labels[idx]==hallu_cls:
                            hallu_idxs.append(idx)
                            file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                            file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                            act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer][head*128:(head*128)+128]).to(device)
                            acts.append(act)
                    acts = torch.stack(acts,axis=0)
                    if 'non_linear_2' in args.method:
                        acts = nlinear_model.relu1(nlinear_model.linear1(inputs)) # pass through model up to classifier
                    elif 'non_linear_4' in args.method: # TODO: to use similarity, add unit norm in forward() before classifier layer
                        pass
                    norm_acts = acts / acts.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                    # Note: with bce, there is only one probe, i.e only one weight vector
                    cur_norm_weights_0 = nlinear_model.classifier.weight / nlinear_model.classifier.weight.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                    sim = torch.sum(norm_acts * cur_norm_weights_0, dim=-1)
                    top_k = torch.topk(sim,args.spl_knn)[1][torch.topk(sim,args.spl_knn)[0]>0].detach().cpu().numpy() # save indices of top k similar vectors (only pos)
                    model_wise_mc_sample_idxs.append(np.array(hallu_idxs)[top_k])
                    print('Similarity of knn samples at current layer:',sim[top_k])
                
                # print(np.array(val_loss))
                if args.save_probes:
                    probe_save_path = f'{args.save_path}/probes/models/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_model{i}_{layer}_{head}'
                    torch.save(nlinear_model, probe_save_path)
                    # probes_saved.append(probe_save_path)
                
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
                            if args.load_act==False:
                                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                            else:
                                act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                            activations.append(act)
                        inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                        predicted = [1 if torch.sigmoid(nlinear_model(inp).data)>0.5 else 0 for inp in inputs] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([1 if torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0]>0.5 else 0 for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                        y_val_pred += predicted
                        y_val_true += batch['labels'].tolist()
                        val_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                        val_preds.append(val_preds_batch)
                        if args.token in ['answer_last','prompt_last','maxpool_all']: val_logits.append(nlinear_model(inputs))
                        if args.token in ['all','tagged_tokens']: val_logits.append(torch.stack([torch.max(nlinear_model(inp).data, dim=0)[0] for inp in inputs]))
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
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file][layer]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, use_prompts[idx], device, args.using_act, layer, args.token, use_answer_token_idxes[idx], use_tagged_token_idxs[idx])
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                            predicted = [1 if torch.sigmoid(nlinear_model(inp).data)>0.5 else 0 for inp in inputs] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([1 if torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0]>0.5 else 0 for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                            y_test_pred += predicted
                            y_test_true += batch['labels'].tolist()
                            test_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                            test_preds.append(test_preds_batch)
                            if args.token in ['answer_last','prompt_last','maxpool_all']: test_logits.append(nlinear_model(inputs))
                            if args.token in ['all','tagged_tokens']: test_logits.append(torch.stack([torch.max(nlinear_model(inp).data, dim=0)[0] for inp in inputs]))
                    all_test_preds[i].append(torch.cat(test_preds).cpu().numpy())
                    all_y_true_test[i].append(y_test_true)
                    all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
                    print('Test F1:',f1_score(y_test_true,y_test_pred),f1_score(y_test_true,y_test_pred,pos_label=0))
                    all_test_logits[i].append(torch.cat(test_logits))
                all_val_logits[i].append(torch.cat(val_logits))
                
        #     break
        # break
    

    # all_val_loss = np.stack([np.stack(all_val_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_loss.npy', all_val_loss)
    # all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_train_loss.npy', all_train_loss)
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_supcon_train_loss.npy', all_supcon_train_loss)
    all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_pred.npy', all_val_preds)
    all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_f1.npy', all_val_f1s)
    all_y_true_val = np.stack([np.array(all_y_true_val[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_true.npy', all_y_true_val)
    all_val_logits = np.stack([torch.stack(all_val_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_val_logits.npy', all_val_logits)
    # all_val_sim = np.stack([np.stack(all_val_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_sim.npy', all_val_sim)
    # all_test_sim = np.stack([np.stack(all_test_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_sim.npy', all_test_sim)

    if args.test_file_name is not None:
        all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_pred.npy', all_test_preds)
        all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_f1.npy', all_test_f1s)
        all_y_true_test = np.stack([np.array(all_y_true_test[i]) for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_true.npy', all_y_true_test)
        all_test_logits = np.stack([torch.stack(all_test_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
        np.save(f'{args.save_path}/probes/NLSC_{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.use_class_wgt}_test_logits.npy', all_test_logits)

if __name__ == '__main__':
    main()