import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_nq, tokenized_mi
import llama
import pickle
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

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

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--token',type=str, default='last')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--file_name", type=str, default=None, help='local directory with dataset')
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
    device = "cuda"

    if args.dataset_name == "tqa_mc2":
        # all_hf_datasets = datasets.list_datasets()
        # print([name for name in all_hf_datasets if 'truthful_qa' in name])
        dataset = load_dataset("truthful_qa", "multiple_choice", streaming= True)['validation']
        print('Here')
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation', streaming= True)['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation', streaming= True)['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'nq': 
        dataset = load_dataset("OamPatel/iti_nq_open_val", streaming= True)['validation']
        formatter = tokenized_nq
    elif args.dataset_name == 'counselling':
        pass
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories, token_idxes = formatter(dataset.with_format('torch'), tokenizer, args.token)
        with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    elif args.dataset_name == 'counselling':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts = tokenized_mi(file_path, tokenizer)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    if 'tqa' in args.dataset_name:
        if 'first' in args.token:
            load_ranges = [(0,1000),(1000,2000),(2000,3000),(3000,3500),(3500,4000),
                            (4000,4500),(4500,5000),(5000,5500),(5500,6000)]
        elif 'answer_all' in args.token:
            load_ranges = [(a*20,(a*20)+20) for a in range(int(500/20)+1)]
        else:
            load_ranges = [(0,1000),(1000,3000),(3000,4000),(4000,5000),(5000,6000)]
    elif 'counselling' in args.dataset_name:
        load_ranges = [(a*20,(a*20)+20) for a in range(int(500/20)+1)] # if ((a*20)+20)>180]
    else:
        load_ranges = [(0,1000),(1000,3000),(3000,5000),(5000,7000),(7000,9000),(9000,11000)]
    for start, end in load_ranges:
        all_layer_wise_activations = []
        all_head_wise_activations = []
        all_mlp_wise_activations = []

        print("Getting activations for "+str(start)+" to "+str(end))
        for prompt,token_idx in tqdm(zip(prompts[start:end],token_idxes[start:end])):
            if args.model_name=='flan_33B':
                layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(base_model, prompt, device)
            else:
                layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt, device)
            if args.token=='last':
                all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                all_head_wise_activations.append(head_wise_activations[:,-1,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,-1,:])
            elif args.token=='first':
                all_layer_wise_activations.append(layer_wise_activations[:,0,:])
                all_head_wise_activations.append(head_wise_activations[:,0,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,0,:])
            elif 'answer_first' in args.token:
                all_layer_wise_activations.append(layer_wise_activations[:,token_idx,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx,:])
            elif 'answer_all' in args.token:
                all_layer_wise_activations.append(layer_wise_activations[:,token_idx:,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx:,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx:,:])

        print("Saving layer wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_layer_wise_{end}.npy', all_layer_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_layer_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_layer_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)
        
        print("Saving head wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_head_wise_{end}.npy', all_head_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_head_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_head_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

        print("Saving mlp wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_mlp_wise_{end}.npy', all_mlp_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_mlp_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_mlp_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

    if 'counselling' not in args.dataset_name:
        print("Saving labels")
        np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_labels_{end}.npy', labels)

if __name__ == '__main__':
    main()