import os
import torch
import torch.nn as nn
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
    return np.sum([b-a for a,b in tagged_token_idxs_prompt])

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
    parser.add_argument('--method',type=str, default='individual_linear')
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_folds',type=int, default=1)
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
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        num_layers = 32
        num_heads = 32
    device = "cuda"

    print("Loading prompts and model responses..")
    if args.dataset_name == 'counselling':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts = tokenized_mi(file_path, tokenizer)
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail':
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

    # Individual probes
    all_train_loss, all_val_loss = {}, {}
    all_val_accs, all_val_f1s = {}, {}
    all_test_accs, all_test_f1s = {}, {}
    all_val_preds, all_test_preds = {}, {}
    y_true_test = {}
    if args.num_folds==1: # Use static test data
        sampled_idxs = np.random.choice(np.arange(1800), size=int(1800*(1-0.2)), replace=False) 
        test_idxs = np.array([x for x in np.arange(1800) if x not in sampled_idxs]) # Sampled indexes from 1800 held-out split
        train_idxs = sampled_idxs if args.len_dataset==1800 else np.arange(args.len_dataset)
    else: # n-fold CV
        fold_idxs = np.array_split(np.arange(args.len_dataset), args.num_folds)
    
    for i in range(args.num_folds):
        print('Training FOLD',i)
        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_folds) if j != i]) if args.num_folds>1 else train_idxs
        test_idxs = fold_idxs[i] if args.num_folds>1 else test_idxs
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-0.2)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        y_train = np.stack([labels[i] for i in train_set_idxs], axis = 0)
        y_val = np.stack([labels[i] for i in val_set_idxs], axis = 0)
        y_test = np.stack([labels[i] for i in test_idxs], axis = 0)
        y_true_test[i] = y_test
        if args.method=='individual_non_linear':
            y_train = np.vstack([[val] for val in y_train], dtype='float32')
            y_val = np.vstack([[val] for val in y_val], dtype='float32')
            y_test = np.vstack([[val] for val in y_test], dtype='float32')

        all_train_loss[i], all_val_loss[i] = [], []
        all_val_accs[i], all_val_f1s[i] = [], []
        all_test_accs[i], all_test_f1s[i] = [], []
        all_val_preds[i], all_test_preds[i] = [], []
        # loop_layers = list(chosen_dims.keys()) if using_chosen_dims else range(num_layers)
        # for layer in tqdm(loop_layers):
        # for layer in tqdm(range(num_layers)):
        for layer in tqdm([0]):
            loop_heads = range(num_heads) if args.using_act == 'ah' else [0]
            for head in loop_heads:
                if args.method=='individual_linear':
                    train_target = np.stack([labels[j] for j in train_set_idxs], axis = 0)
                    class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                    weight = 1. / class_sample_count
                    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    ds_train = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": y_train}).with_format("torch")
                    ds_train = DataLoader(ds_train, batch_size=128,sampler=sampler)
                    ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                    ds_val = DataLoader(ds_val, batch_size=128)
                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                    ds_test = DataLoader(ds_test, batch_size=128)

                    act_dims = {'mlp':4096,'mlp_l1':11008,'ah':128}
                    linear_model = LogisticRegression_Torch(act_dims[args.using_act], 2).to(device)
                    criterion = nn.BCELoss()
                    lr = 0.05
                    
                    # iter_bar = tqdm(ds_train, desc='Train Iter (loss=X.XXX)')

                    train_loss = []
                    for epoch in range(10):
                        linear_model.train()
                        optimizer = torch.optim.SGD(linear_model.parameters(), lr=lr)
                        # for step,batch in enumerate(iter_bar):
                        for step,batch in enumerate(ds_train):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                activations.append(get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx]))
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)
                            if args.token in ['answer_last','prompt_last','maxpool_all']:
                                targets = batch['labels']
                            elif args.token=='all':
                                # print(step,prompt_tokens[idx],len(prompt_tokens[idx]))
                                targets = torch.cat([torch.Tensor([y_label for j in range(len(prompt_tokens[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0)
                            if args.token=='tagged_all':
                                targets = torch.cat([torch.Tensor([y_label for j in range(num_tagged_tokens(tagged_token_idxs[idx]))]) for idx,y_label in zip(batch['inputs_idxs'],batch['labels'])],dim=0)
                            outputs = linear_model(inputs)
                            loss = criterion(outputs, nn.functional.one_hot(targets.to(torch.int64),num_classes=2).to(torch.float32).to(device))
                            train_loss.append(loss.item())
                            # iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
                            loss.backward()
                            optimizer.step()
                        lr = lr*0.9
                    all_train_loss[i].append(np.array(train_loss))
                    pred_correct = 0
                    y_val_pred, y_val_true = [], []
                    val_preds = []
                    with torch.no_grad():
                        linear_model.eval()
                        for step,batch in enumerate(ds_val):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                activations.append(get_llama_activations_bau_custom(model, tokenized_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx]))
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                            predicted = torch.max(linear_model(inputs).data, dim=1)[1] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.max(linear_model(inp).data, dim=0)[0], dim=0)[1] for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                            # pred_correct += (predicted == batch['labels'].to(device)).sum()
                            y_val_pred += predicted.cpu().tolist()
                            y_val_true += batch['labels'].tolist()
                            val_preds += torch.max(linear_model(inputs).data, dim=1)[0] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(linear_model(inp).data, dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                    all_val_preds[i].append(val_preds.numpy())
                    # print('Validation Acc:',pred_correct/len(X_val))
                    # all_val_accs[i].append(pred_correct/len(X_val))
                    all_val_f1s[i].append(f1_score(y_val_true,y_val_pred))
                    pred_correct = 0
                    y_test_pred, y_test_true = [], []
                    test_preds = []
                    with torch.no_grad():
                        linear_model.eval()
                        use_prompts = tokenized_prompts if args.num_folds>1 else test_tokenized_prompts
                        for step,batch in enumerate(ds_test):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                activations.append(get_llama_activations_bau_custom(model, use_prompts[idx], device, args.using_act, layer, args.token, answer_token_idxes[idx], tagged_token_idxs[idx]))
                            inputs = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else activations
                            predicted = torch.max(linear_model(inputs).data, dim=1)[1] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(torch.max(linear_model(inp).data, dim=0)[0], dim=0)[1] for inp in inputs]) # For each sample, get max prob per class across tokens, then choose the class with highest prob
                            # pred_correct += (predicted == batch['labels'].to(device)).sum()
                            y_test_pred += predicted.cpu().tolist()
                            y_test_true += batch['labels'].tolist()
                            test_preds += torch.max(linear_model(inputs).data, dim=1)[0] if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.stack([torch.max(linear_model(inp).data, dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                    all_test_preds[i].append(test_preds.numpy())
                    # print('Test Acc:',pred_correct/len(X_test))
                    # all_test_accs[i].append(pred_correct/len(X_test))
                    all_test_f1s[i].append(f1_score(y_test_true,y_test_pred))
    all_train_loss = np.stack([np.stack(all_train_loss[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_train_loss.npy', all_train_loss)
    all_val_preds = np.stack([np.stack(all_val_preds[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_val_pred.npy', all_val_preds)
    all_test_preds = np.stack([np.stack(all_test_preds[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_test_pred.npy', all_test_preds)
    all_val_f1s = np.stack([np.array(all_val_f1s[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_val_f1.npy', all_val_f1s)
    all_test_f1s = np.stack([np.array(all_test_f1s[i]) for i in range(args.num_folds)])
    np.save(f'{args.save_path}/probes/{args.model_name}_{args.train_file_name}_{args.len_dataset}_{args.num_folds}_{args.using_act}_{args.token}_{args.method}_test_f1.npy', all_test_f1s)
    

if __name__ == '__main__':
    main()