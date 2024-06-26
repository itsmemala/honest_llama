import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import statistics
import pickle
import json
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, get_token_tags
from utils import LogisticRegression_Torch, FeedforwardNeuralNetModel
from copy import deepcopy
import llama
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
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

def get_logits(ds_train_fixed,layer,head,linear_model,device,args):
    logits = []
    linear_model.eval()
    for step,batch in enumerate(ds_train_fixed):
        activations = []
        for idx in batch['inputs_idxs']:
            if args.load_act==False:
                act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                file_end = idx-(idx%100)+100 # 487: 487-(87)+100
                file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
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
        logits.append(linear_model(inputs))
    return logits


def train_classifier_on_probes(train_logits,y_train,val_logits,y_val,test_logits,y_test,sampler,device,args):
    
    # print(train_logits.shape)
    ds_train = Dataset.from_dict({"inputs": train_logits, "labels": y_train}).with_format("torch")
    ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler)
    ds_val = Dataset.from_dict({"inputs": val_logits, "labels": y_val}).with_format("torch")
    ds_val = DataLoader(ds_val, batch_size=args.bs)
    ds_test = Dataset.from_dict({"inputs": test_logits, "labels": y_test}).with_format("torch")
    ds_test = DataLoader(ds_test, batch_size=args.bs)

    linear_model = LogisticRegression_Torch(train_logits.shape[1], 2).to(device)
    wgt_0 = np.sum(y_train)/len(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.CrossEntropyLoss()
    lr = args.lr
    
    # iter_bar = tqdm(ds_train, desc='Train Iter (loss=X.XXX)')

    val_loss = []
    best_val_loss = torch.inf
    best_model_state = deepcopy(linear_model.state_dict())
    if args.optimizer=='Adam_w_lr_sch' or args.optimizer=='SGD_w_lr_sch':
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr) if 'Adam' in args.optimizer else torch.optim.SGD(linear_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        linear_model.train()
        if args.optimizer=='Adam' or args.optimizer=='SGD': optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr) if 'Adam' in args.optimizer else torch.optim.SGD(linear_model.parameters(), lr=lr)
        for step,batch in enumerate(ds_train):
            optimizer.zero_grad()
            outputs = linear_model(batch['inputs'].to(device))
            loss = criterion(outputs, batch['labels'].to(device))
            loss.backward()
            optimizer.step()
        # Get val loss
        linear_model.eval()
        epoch_val_loss = 0
        for step,batch in enumerate(ds_val):
            optimizer.zero_grad()
            outputs = linear_model(batch['inputs'].to(device))
            epoch_val_loss += criterion(outputs, batch['labels'].to(device))
        val_loss.append(epoch_val_loss.item())
        # Choose best model
        if epoch_val_loss.item() < best_val_loss:
            best_val_loss = epoch_val_loss.item()
            best_model_state = deepcopy(linear_model.state_dict())
        # Early stopping
        patience, min_val_loss_drop, is_not_decreasing = 5, 1, 0
        if len(val_loss)>=patience:
            for epoch_id in range(1,patience,1):
                val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
            if is_not_decreasing==patience-1: break
        if args.optimizer=='SGD': lr = lr*0.75 # No decay for Adam
        if args.optimizer=='Adam_w_lr_sch' or args.optimizer=='SGD_w_lr_sch': scheduler.step()
    linear_model.load_state_dict(best_model_state)
    
    # Val and Test performance
    y_val_pred, y_val_true = [], []
    pred_correct = 0
    with torch.no_grad():
        linear_model.eval()
        for step,batch in enumerate(ds_val):
            predicted = torch.max(linear_model(batch['inputs'].to(device)).data, dim=1)[1]
            y_val_pred += predicted.cpu().tolist()
            y_val_true += batch['labels'].to(device).tolist()
    print('Val F1 (logits):',f1_score(y_val_true,y_val_pred),f1_score(y_val_true,y_val_pred,pos_label=0))
    y_test_pred, y_test_true = [], []
    pred_correct = 0
    with torch.no_grad():
        linear_model.eval()
        for step,batch in enumerate(ds_test):
            predicted = torch.max(linear_model(batch['inputs'].to(device)).data, dim=1)[1]
            y_test_pred += predicted.cpu().tolist()
            y_test_true += batch['labels'].to(device).tolist()
    print('Test F1 (logits):',f1_score(y_test_true,y_test_pred),f1_score(y_test_true,y_test_pred,pos_label=0))

    return

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
    parser.add_argument('--method',type=str, default='individual_linear') # individual_linear_unitnorm, individual_linear, individual_linear_kld, individual_linear_kld_reverse
    parser.add_argument('--kld_wgt',type=float, default=1)
    parser.add_argument('--classifier_on_probes',type=bool, default=False)
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
    parser.add_argument('--bs',type=int, default=4)
    parser.add_argument('--epochs',type=int, default=3)
    parser.add_argument('--layer_start',type=int, default=None)
    parser.add_argument('--layer_end',type=int, default=None)
    parser.add_argument('--lr',type=float, default=0.05)
    parser.add_argument('--optimizer',type=str, default='SGD')
    parser.add_argument('--use_class_wgt',type=bool, default=False)
    parser.add_argument('--load_act',type=bool, default=False)
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
        num_layers = 60
        num_heads = 52
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
                labels.append(1 if data['rouge1_to_target']>0.3 else 0)
        labels = labels[:args.len_dataset]
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer)
        test_labels = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json', 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                test_labels.append(1 if data['rouge1_to_target']>0.3 else 0)
    
    if args.token=='tagged_tokens':
        tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
        test_tagged_token_idxs = get_token_tags(test_prompts,test_prompt_tokens)
    else:
        tagged_token_idxs,test_tagged_token_idxs = [[] for i in range(len(prompts))],[[] for i in range(len(test_prompts))]
    
    # Probe training
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    # Individual probes
    all_train_loss, all_val_loss = {}, {}
    all_val_accs, all_val_f1s = {}, {}
    all_test_accs, all_test_f1s = {}, {}
    all_val_preds, all_test_preds = {}, {}
    all_y_true_val, all_y_true_test = {}, {}
    all_train_logits, all_val_logits, all_test_logits = {}, {}, {}
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
    
    method_concat = args.method + '_' + str(args.kld_wgt) if 'kld' in args.method else args.method

    for i in range(args.num_folds):
        print('Training FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        y_train = np.stack([labels[i] for i in train_set_idxs], axis = 0)
        y_val = np.stack([labels[i] for i in val_set_idxs], axis = 0)
        y_test = np.stack([labels[i] for i in test_idxs], axis = 0) if args.num_folds>1 else np.stack([test_labels[i] for i in test_idxs], axis = 0)
        # if args.method=='individual_non_linear':
        #     y_train = np.vstack([[val] for val in y_train], dtype='float32')
        #     y_val = np.vstack([[val] for val in y_val], dtype='float32')
        #     y_test = np.vstack([[val] for val in y_test], dtype='float32')

        all_train_loss[i], all_val_loss[i] = [], []
        all_val_accs[i], all_val_f1s[i] = [], []
        all_test_accs[i], all_test_f1s[i] = [], []
        all_val_preds[i], all_test_preds[i] = [], []
        all_y_true_val[i], all_y_true_test[i] = [], []
        all_train_logits[i], all_val_logits[i], all_test_logits[i] = [], [], []
        all_val_sim[i], all_test_sim[i] = [], []
        probes_saved = []
        loop_layers = range(args.layer_start,args.layer_end+1,1) if args.layer_start is not None else range(num_layers)
        if args.method=='individual_linear_kld_reverse': loop_layers = range(num_layers-1,-1,-1)
        # for layer in tqdm(loop_layers):
        # for layer in tqdm(range(num_layers)):
        for layer_iterator,layer in tqdm(enumerate([31,31])):
            loop_heads = range(num_heads) if args.using_act == 'ah' else [0]
            for head in loop_heads:
                if args.method=='individual_linear' or args.method=='individual_linear_unitnorm' or args.method=='individual_linear_kld' or args.method=='individual_linear_kld_reverse':
                    train_target = np.stack([labels[j] for j in train_set_idxs], axis = 0)
                    class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                    weight = 1. / class_sample_count
                    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    ds_train = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train}).with_format("torch")
                    ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler)
                    ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                    ds_val = DataLoader(ds_val, batch_size=args.bs)
                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                    ds_test = DataLoader(ds_test, batch_size=args.bs)

                    act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128,'layer':4096}
                    linear_model = LogisticRegression_Torch(act_dims[args.using_act], 2).to(device)
                    wgt_0 = np.sum(y_train)/len(y_train)
                    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.CrossEntropyLoss()
                    criterion_kld = nn.KLDivLoss(reduction='batchmean')
                    lr = args.lr
                    
                    # iter_bar = tqdm(ds_train, desc='Train Iter (loss=X.XXX)')

                    train_loss, val_loss, kld_loss = [], [], []
                    best_val_loss = torch.inf
                    best_model_state = linear_model.state_dict()
                    best_val_logits = []
                    if args.optimizer=='Adam_w_lr_sch' or args.optimizer=='SGD_w_lr_sch':
                        optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr) if 'Adam' in args.optimizer else torch.optim.SGD(linear_model.parameters(), lr=lr)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                    for epoch in range(args.epochs):
                        linear_model.train()
                        if args.optimizer=='Adam' or args.optimizer=='SGD': optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr) if 'Adam' in args.optimizer else torch.optim.SGD(linear_model.parameters(), lr=lr)
                        # for step,batch in enumerate(iter_bar):
                        for step,batch in enumerate(ds_train):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%100)+100 # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx]) # TODO for AH: extract specific head activations
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
                            if args.method=='individual_linear_unitnorm': inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                            outputs = linear_model(inputs)
                            loss = criterion(outputs, targets.to(device))
                            if 'individual_linear_kld' in args.method and len(probes_saved)>0:
                                train_preds_batch = F.softmax(linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                                for probes_saved_path in probes_saved:
                                    past_linear_model = LogisticRegression_Torch(act_dims[args.using_act], 2).to(device)
                                    past_linear_model = torch.load(probes_saved_path)
                                    past_preds_batch = F.softmax(past_linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(past_linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                                    loss = loss + args.kld_wgt/criterion_kld(train_preds_batch[:,0],past_preds_batch[:,0])
                                    kld_loss.append(1/criterion_kld(train_preds_batch[:,0],past_preds_batch[:,0]).item())
                            train_loss.append(loss.item())
                            # iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
                            loss.backward()
                            optimizer.step()
                            # if 'individual_linear_kld' in args.method and len(probes_saved)>0:
                            #     print('Total loss:',loss.item())
                            #     print('KLD loss:',1/criterion_kld(train_preds_batch[:,0],past_preds_batch[:,0]).item())
                        # Get val loss
                        linear_model.eval()
                        epoch_val_loss = 0
                        epoch_val_logits = []
                        for step,batch in enumerate(ds_val):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%100)+100 # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx]) # TODO for AH: extract specific head activations
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
                            if args.method=='individual_linear_unitnorm': inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                            outputs = linear_model(inputs)
                            epoch_val_loss += criterion(outputs, targets.to(device))
                            epoch_val_logits.append(outputs)
                        val_loss.append(epoch_val_loss.item())
                        # Choose best model
                        if epoch_val_loss.item() < best_val_loss:
                            best_val_loss = epoch_val_loss.item()
                            best_model_state = linear_model.state_dict()
                            best_val_logits = epoch_val_logits
                        # Early stopping
                        patience, min_val_loss_drop, is_not_decreasing = 5, 1, 0
                        if len(val_loss)>=patience:
                            for epoch_id in range(1,patience,1):
                                val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                                if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                            if is_not_decreasing==patience-1: break
                        if args.optimizer=='SGD': lr = lr*0.75 # No decay for Adam
                        if args.optimizer=='Adam_w_lr_sch' or args.optimizer=='SGD_w_lr_sch': scheduler.step()
                    all_train_loss[i].append(np.array(train_loss))
                    all_val_loss[i].append(np.array(val_loss))
                    if 'individual_linear_kld' in args.method and len(probes_saved)>0:
                        print(layer,head)
                        print('KLD loss:',kld_loss[:10],kld_loss[-10:])
                        print('Train and val loss:',train_loss[-1],val_loss[-1])
                        print('\n')
                    linear_model.load_state_dict(best_model_state)
                    if args.save_probes:
                        pass
                        # probe_save_path = f'{args.save_path}/probes/models/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_model{i}_{layer}_{head}'
                        # torch.save(linear_model, probe_save_path)
                        # probes_saved.append(probe_save_path)
                    if layer_iterator==0:
                        probe_wgts_first_cls0, probe_wgts_first_cls1 = linear_model.linear.weight[0], linear_model.linear.weight[1]
                    if layer_iterator==1:
                        probe_wgts_second_cls0, probe_wgts_second_cls1 = linear_model.linear.weight[0], linear_model.linear.weight[1]
                        norm_weights_a0 = probe_wgts_first_cls0 / probe_wgts_first_cls0.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        norm_weights_a1 = probe_wgts_first_cls1 / probe_wgts_first_cls1.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        norm_weights_b0 = probe_wgts_second_cls0 / probe_wgts_second_cls0.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        norm_weights_b1 = probe_wgts_second_cls1 / probe_wgts_second_cls1.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        print('Probe similarity on same layer:',torch.sum(norm_weights_a0*norm_weights_b0).item(),torch.sum(norm_weights_a1*norm_weights_b1).item())
                    if args.classifier_on_probes:
                        # Fix train order and get logits
                        ds_train_fixed = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train}).with_format("torch")
                        ds_train_fixed = DataLoader(ds_train_fixed, batch_size=args.bs)
                        best_train_logits = get_logits(ds_train_fixed,layer,head,linear_model,device,args)
                    
                    # Val and Test performance
                    pred_correct = 0
                    y_val_pred, y_val_true = [], []
                    val_preds = []
                    val_sim = []
                    with torch.no_grad():
                        linear_model.eval()
                        norm_weights_0 = linear_model.linear.weight[0] / linear_model.linear.weight[0].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        norm_weights_1 = linear_model.linear.weight[1] / linear_model.linear.weight[1].pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # unit normalise
                        for step,batch in enumerate(ds_val):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%100)+100 # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx])
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                            if args.method=='individual_linear_unitnorm': inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                            predicted = torch.max(linear_model(inputs).data, dim=1)[1] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.max(linear_model(inp).data, dim=0)[0], dim=0)[1] for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                            y_val_pred += predicted.cpu().tolist()
                            y_val_true += batch['labels'].tolist()
                            val_preds_batch = F.softmax(linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                            val_preds.append(val_preds_batch)
                            val_sim.append(torch.stack((torch.sum(inputs * norm_weights_0.detach(), dim=-1),torch.sum(inputs * norm_weights_1.detach(), dim=-1)),dim=1)) # + linear_model.linear.bias
                    all_val_preds[i].append(torch.cat(val_preds).cpu().numpy())
                    all_val_sim[i].append(torch.cat(val_sim).cpu().numpy())
                    all_y_true_val[i].append(y_val_true)
                    all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
                    pred_correct = 0
                    y_test_pred, y_test_true = [], []
                    test_preds = []
                    test_logits = []
                    test_sim = []
                    with torch.no_grad():
                        linear_model.eval()
                        use_prompts = tokenized_prompts if args.num_folds>1 else test_tokenized_prompts
                        use_answer_token_idxes = answer_token_idxes if args.num_folds>1 else test_answer_token_idxes
                        use_tagged_token_idxs = tagged_token_idxs if args.num_folds>1 else test_tagged_token_idxs
                        for step,batch in enumerate(ds_test):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                if args.load_act==False:
                                    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}
                                    file_end = idx-(idx%100)+100 # 487: 487-(87)+100
                                    file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
                                    act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer]).to(device) if 'mlp' in args.using_act or 'layer' in args.using_act else torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%100][layer][head*128:(head*128)+128]).to(device)
                                else:
                                    act = get_llama_activations_bau_custom(model, use_prompts[idx], device, args.using_act, layer, args.token, use_answer_token_idxes[idx], use_tagged_token_idxs[idx])
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                            if args.method=='individual_linear_unitnorm': inputs = inputs / inputs.pow(2).sum(dim=1).sqrt().unsqueeze(-1) # unit normalise
                            predicted = torch.max(linear_model(inputs).data, dim=1)[1] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.max(linear_model(inp).data, dim=0)[0], dim=0)[1] for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                            y_test_pred += predicted.cpu().tolist()
                            y_test_true += batch['labels'].tolist()
                            test_preds_batch = F.softmax(linear_model(inputs).data, dim=1) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(F.softmax(linear_model(inp).data, dim=1), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                            test_preds.append(test_preds_batch)
                            test_logits.append(linear_model(inputs))
                            test_sim.append(torch.stack((torch.sum(inputs * norm_weights_0.detach(), dim=-1),torch.sum(inputs * norm_weights_1.detach(), dim=-1)),dim=1)) # + linear_model.linear.bias
                    all_test_preds[i].append(torch.cat(test_preds).cpu().numpy())
                    all_test_sim[i].append(torch.cat(test_sim).cpu().numpy())
                    all_y_true_test[i].append(y_test_true)
                    all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
                    if args.classifier_on_probes:
                        all_train_logits[i].append(torch.cat(best_train_logits))
                    all_val_logits[i].append(torch.cat(best_val_logits))
                    all_test_logits[i].append(torch.cat(test_logits))
            #     break
            # break
    
        if args.classifier_on_probes:
            train_logits = torch.cat(all_train_logits[i],dim=1)
            val_logits = torch.cat(all_val_logits[i],dim=1)
            test_logits = torch.cat(all_test_logits[i],dim=1)
            train_classifier_on_probes(train_logits,y_train,val_logits,y_val,test_logits,y_test,sampler,device,args)

    # # all_val_loss = np.stack([np.stack(all_val_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_loss.npy', all_val_loss)
    # # all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)]) # Can only stack if number of epochs is same for each probe
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_train_loss.npy', all_train_loss)
    # all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_pred.npy', all_val_preds)
    # all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_pred.npy', all_test_preds)
    # all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_f1.npy', all_val_f1s)
    # all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_f1.npy', all_test_f1s)
    # all_y_true_val = np.stack([np.array(all_y_true_val[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_true.npy', all_y_true_val)
    # all_y_true_test = np.stack([np.array(all_y_true_test[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_true.npy', all_y_true_test)
    # all_val_logits = np.stack([torch.stack(all_val_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_logits.npy', all_val_logits)
    # all_test_logits = np.stack([torch.stack(all_test_logits[i]).detach().cpu().numpy() for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_logits.npy', all_test_logits)
    # all_val_sim = np.stack([np.stack(all_val_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_val_sim.npy', all_val_sim)
    # all_test_sim = np.stack([np.stack(all_test_sim[i]) for i in range(args.num_folds)])
    # np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{method_concat}_bs{args.bs}_epochs{args.epochs}_{args.lr}_{args.optimizer}_{args.use_class_wgt}_{args.layer_start}_{args.layer_end}_test_sim.npy', all_test_sim)

if __name__ == '__main__':
    main()