import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, get_llama_activations_bau_custom, get_token_tags, get_token_nll
from utils import tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_nq, tokenized_mi, tokenized_mi_v2, tokenized_from_file, tokenized_from_file_v2
import llama
import pickle
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

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

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--act_type',type=str, default='during')
    parser.add_argument('--token',type=str, default='last')
    parser.add_argument('--mlp_l1',type=str, default='No')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--num_samples',type=int, default=None)
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
    num_layers = 33 if '7B' in args.model_name and args.token=='tagged_tokens' else 32 if '7B' in args.model_name else 40 if '13B' in args.model_name else 60 if '33B' in args.model_name else 0 #raise ValueError("Unknown model size.")

    # if args.dataset_name == "tqa_mc2":
    #     # all_hf_datasets = datasets.list_datasets()
    #     # print([name for name in all_hf_datasets if 'truthful_qa' in name])
    #     dataset = load_dataset("truthful_qa", "multiple_choice", streaming= True)['validation']
    #     print('Here')
    #     formatter = tokenized_tqa
    # elif args.dataset_name == "tqa_gen": 
    #     dataset = load_dataset("truthful_qa", 'generation', streaming= True)['validation']
    #     formatter = tokenized_tqa_gen
    # elif args.dataset_name == 'tqa_gen_end_q': 
    #     dataset = load_dataset("truthful_qa", 'generation', streaming= True)['validation']
    #     formatter = tokenized_tqa_gen_end_q
    # elif args.dataset_name == 'nq': 
    #     dataset = load_dataset("OamPatel/iti_nq_open_val", streaming= True)['validation']
    #     formatter = tokenized_nq
    # elif args.dataset_name == 'counselling' or args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name == 'strqa' or args.dataset_name == 'gsm8k':
    #     pass
    # else: 
    #     raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen": # or args.dataset_name == "tqa_gen_end_q": 
        # prompts, labels, categories, token_idxes = formatter(dataset.with_format('torch'), tokenizer, args.token)
        # if len(token_idxes)==0:
        #     token_idxes = [-1 for i in prompts]
        # with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
        #     pickle.dump(categories, f)
        file_path = f'{args.save_path}/responses/{args.file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.file_name}_response_start_token_idx.npy', answer_token_idxes)
    elif args.dataset_name == 'counselling':
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts = tokenized_mi(file_path, tokenizer)
    elif args.dataset_name == 'counselling_wudata':
        file_path = f'{args.save_path}/responses/'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_mi_v2(file_path, tokenizer)
    elif args.dataset_name == 'strqa' or args.dataset_name == 'gsm8k' or ('baseline' in args.file_name or 'dola' in args.file_name):
        num_samples = args.num_samples if ('sampled' in args.file_name and args.num_samples is not None) else 8 if 'sampled' in args.file_name else 1
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer, num_samples)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.file_name}_response_start_token_idx.npy', answer_token_idxes)
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa':
        num_samples = args.num_samples if ('sampled' in args.file_name and args.num_samples is not None) else 10 if 'sampled' in args.file_name else 1
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, num_samples)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.file_name}_response_start_token_idx.npy', answer_token_idxes)
    # else: 
    #     prompts, labels = formatter(dataset, tokenizer)

    if 'tagged_tokens' in args.token:
        tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
    else:
        tagged_token_idxs = [() for prompt in prompts] # Saves time


    if 'tqa' in args.dataset_name:
        # if args.token=='last' or args.token=='answer_first':
        #     # load_ranges = [(0,1000),(1000,2000),(2000,3000),(3000,3500),(3500,4000),
        #     #                 (4000,4500),(4500,5000),(5000,5500),(5500,6000)] # llama-13B? first?
        #     load_ranges = [(a*100,(a*100)+100) for a in range(int(6000/100))]
        # elif 'answer_all' in args.token:
        #     load_ranges = [(a*20,(a*20)+20) for a in range(int(500/20)+1)]
        # else:
        #     load_ranges = [(0,1000),(1000,3000),(3000,4000),(4000,5000),(5000,6000)]
        if 'train' in args.file_name:
            load_ranges = [(a*100,(a*100)+100) for a in range(int(4800/100))]
        else:
            load_ranges = [(a*100,(a*100)+100) for a in range(int(1300/100))]
    elif 'counselling' in args.dataset_name:
        load_ranges = [(a*20,(a*20)+20) for a in range(int(500/20)+1)] # if ((a*20)+20)>180]
    elif args.dataset_name=='nq':
        load_ranges = [(0,1000),(1000,3000),(3000,5000),(5000,7000),(7000,9000),(9000,11000)]
    elif args.dataset_name=='nq_open':
        if args.token=='all':
            # load_ranges = [(a*5,(a*5)+5) for a in range(int(1800/5)+1) if (a*5)+5>860] # 20 upto 540, 10 upto 860, 5 upto 1800
            # load_ranges = [(1380,1385)]
            # load_ranges = [(a*20,(a*20)+20) for a in range(int(1800/20)+1) if (a*20)+20>1700] # mlp_l1
            load_ranges = [(a*100,(a*100)+100) for a in range(int(1800/100))] # alpaca7B
        else:
            if '5000' in args.file_name:
                load_ranges = [(a*100,(a*100)+100) for a in range(int(5000/100))] # train file
            else:
                load_ranges = [(a*100,(a*100)+100) for a in range(int(1800/100))] # test file
                # load_ranges = [(a*50,(a*50)+50) for a in range(int(1800/50))] # test file (only for hallucheck3)
    elif args.dataset_name=='cnn_dailymail':
        if args.token=='prompt_last_onwards':
            load_ranges = [(a*20,(a*20)+20) for a in range(int(1000/20)+1) if (a*20)+20>520]
        else:
            load_ranges = [(0,1000)]
    elif args.dataset_name == 'trivia_qa':
        # load_ranges = [(a*100,(a*100)+100) for a in range(int(5000/100)) if (a*100)+100>1800] # dola generation file
        # if '5000' in args.file_name: # or 'train' in args.file_name:
            # load_ranges = [(a*100,(a*100)+100) for a in range(int(5000/100))] # train file
        # elif '20000' in args.file_name:
        #     load_ranges = [(a*100,(a*100)+100) for a in range(int(20000/100))] # train file
        # elif '2000' in args.file_name:
        #     load_ranges = [(a*100,(a*100)+100) for a in range(int(2000/100))] # test file
        # else:
        #     load_ranges = [(a*100,(a*100)+100) for a in range(int(1800/100))] # test file
        load_ranges = [(a*100,(a*100)+100) for a in range(int(len(prompts)/100))] # train file with sampled responses
    elif args.dataset_name == 'strqa':
        # load_ranges = [(a*50,(a*50)+50) for a in range(int(2300/50))] # all responses
        # print(len(prompts))
        load_ranges = [(a*50,(a*50)+50) for a in range(int(len(prompts)/50)+1)] # if (a*50)+50 > 14600]
        # load_ranges = [(16450,16500)]
        # print(load_ranges[-1])
    elif args.dataset_name == 'gsm8k':
        load_ranges = [(a*20,(a*20)+20) for a in range(int(1400/20))] # all responses

    # load_ranges = [(0,len(prompts))]
    
    for start, end in load_ranges:
        all_layer_wise_activations = []
        all_head_wise_activations = []
        all_mlp_wise_activations = []
        all_token_logprobs = []

        print("Getting activations for "+str(start)+" to "+str(end))
        for prompt,token_idx,tagged_idxs in tqdm(zip(tokenized_prompts[start:end],answer_token_idxes[start:end],tagged_token_idxs[start:end])):
            if args.mlp_l1=='Yes':
                mlp_wise_activations = get_llama_activations_bau(model, prompt, device, mlp_l1=args.mlp_l1)
                if args.token=='answer_last': #last
                    all_mlp_wise_activations.append(mlp_wise_activations[:,-1,:])
                elif args.token=='prompt_last':
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1,:])
                elif args.token=='maxpool_all':
                    all_mlp_wise_activations.append(np.max(mlp_wise_activations,axis=1))
                elif 'answer_all' in args.token:
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx:,:])
                elif args.token=='all':
                    all_mlp_wise_activations.append(mlp_wise_activations[:,:,:])
                elif args.token=='prompt_last_onwards':
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1:,:])
            else:
                if args.model_name=='flan_33B':
                    layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(base_model, prompt, device)
                else:
                    layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt, device)
                if args.token=='answer_last': #last
                    all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                    all_head_wise_activations.append(head_wise_activations[:,-1,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,-1,:])
                elif args.token=='slt': #last
                    all_layer_wise_activations.append(layer_wise_activations[:,-2,:])
                    all_head_wise_activations.append(head_wise_activations[:,-2,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,-2,:])
                elif args.token=='prompt_last':
                    all_layer_wise_activations.append(layer_wise_activations[:,token_idx-1,:])
                    all_head_wise_activations.append(head_wise_activations[:,token_idx-1,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1,:])
                elif args.token=='least_likely':
                    # print(print(tokenizer.decode(prompt[0], skip_special_tokens=True)))
                    # print(tokenizer.decode(prompt[0], skip_special_tokens=True))
                    least_likely_nll, least_likely_token_idx = 0, token_idx-1
                    for next_token_idx in range(len(prompt[0][token_idx:])):
                        predicting_token_idx = token_idx+next_token_idx-1 # -1 since prob of every next token is given by prev token
                        predicted_token_id = prompt[0][token_idx+next_token_idx]
                        part_prompt = prompt[:,:predicting_token_idx]
                        # print(tokenizer.decode(part_prompt, skip_special_tokens=True))
                        nll = get_token_nll(model, part_prompt, device, predicted_token_id)
                        if nll > least_likely_nll:
                            least_likely_nll = nll
                            least_likely_token_idx = predicting_token_idx
                    act = get_llama_activations_bau_custom(model, prompt, device, 'layer', -1, args.token, least_likely_token_idx)
                    all_layer_wise_activations.append(act.numpy())
                elif args.token=='random':
                    # if len(prompt[0][token_idx:])==0: print(tokenizer.decode(prompt[0], skip_special_tokens=True))
                    random_token_idx = token_idx-1 + np.random.choice(len(prompt[0][token_idx-1:]), 1)
                    act = get_llama_activations_bau_custom(model, prompt, device, 'layer', -1, args.token, random_token_idx)
                    all_layer_wise_activations.append(act.numpy())
                elif args.token=='prompt_last_and_answer_last':
                    all_layer_wise_activations.append(np.stack((layer_wise_activations[:,token_idx-1,:],layer_wise_activations[:,-1,:]),axis=1))
                    all_head_wise_activations.append(np.stack((head_wise_activations[:,token_idx-1,:],head_wise_activations[:,-1,:]),axis=1))
                    all_mlp_wise_activations.append(np.stack((mlp_wise_activations[:,token_idx-1,:],mlp_wise_activations[:,-1,:]),axis=1))
                elif args.token=='maxpool_all':
                    all_layer_wise_activations.append(np.max(layer_wise_activations,axis=1))
                    all_head_wise_activations.append(np.max(head_wise_activations,axis=1))
                    all_mlp_wise_activations.append(np.max(mlp_wise_activations,axis=1))
                elif 'answer_first' in args.token:
                    all_layer_wise_activations.append(layer_wise_activations[:,token_idx,:])
                    all_head_wise_activations.append(head_wise_activations[:,token_idx,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx,:])
                elif 'answer_all' in args.token:
                    all_layer_wise_activations.append(layer_wise_activations[:,token_idx:,:])
                    all_head_wise_activations.append(head_wise_activations[:,token_idx:,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx:,:])
                elif args.token=='all':
                    all_layer_wise_activations.append(layer_wise_activations[:,:,:])
                    all_head_wise_activations.append(head_wise_activations[:,:,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,:,:])
                elif args.token=='prompt_last_onwards':
                    # all_layer_wise_activations.append(layer_wise_activations[:,:,:])
                    all_head_wise_activations.append(head_wise_activations[:,token_idx-1:,:])
                    all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1:,:])
                elif args.token=='tagged_tokens' or args.token=='tagged_tokens_and_last':
                    acts = []
                    for layer in range(num_layers):
                        act = get_llama_activations_bau_custom(model, prompt, device, 'layer', layer, args.token, token_idx, tagged_idxs)
                        acts.append(act)
                    # print(len(acts),acts[0].shape)
                    acts = torch.stack(acts)
                    all_layer_wise_activations.append(acts)
            # token_logprobs = []
            # for next_token_idx in range(len(prompt[0][token_idx:])):
            #     predicting_token_idx = token_idx+next_token_idx-1 # -1 since prob of every next token is given by prev token
            #     predicted_token_id = prompt[0][token_idx+next_token_idx]
            #     part_prompt = prompt[:,:predicting_token_idx]
            #     # print(tokenizer.decode(part_prompt, skip_special_tokens=True))
            #     token_logprobs.append(-get_token_nll(model, part_prompt, device, predicted_token_id)) # apply neg to match sign returned by openai API for token logprobs
            # all_token_logprobs.append(token_logprobs)
        
        #     break
        # break

        if args.mlp_l1=='Yes':
            print("Saving mlp l1 activations")
            with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_mlp_l1_{end}.pkl', 'wb') as outfile:
                pickle.dump(all_mlp_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)
        else:
            print("Saving layer wise activations")
            if 'tagged_tokens' in args.token:
                with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_layer_wise_{end}.pkl', 'wb') as outfile:
                    torch.save(all_layer_wise_activations, outfile, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_layer_wise_{end}.npy', all_layer_wise_activations)
                with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_layer_wise_{end}.pkl', 'wb') as outfile:
                    pickle.dump(all_layer_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)
            
            print("Saving head wise activations")
            # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_head_wise_{end}.npy', all_head_wise_activations)
            with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_head_wise_{end}.pkl', 'wb') as outfile:
                pickle.dump(all_head_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

            print("Saving mlp wise activations")
            # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_mlp_wise_{end}.npy', all_mlp_wise_activations)
            with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_mlp_wise_{end}.pkl', 'wb') as outfile:
                pickle.dump(all_mlp_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

        # with open(f'{args.save_path}/features/counselling_wudata_token_logprobs.pkl', 'wb') as outfile:
        #     pickle.dump(all_token_logprobs, outfile, pickle.HIGHEST_PROTOCOL)

    # if 'counselling' not in args.dataset_name and args.dataset_name!='nq_open' and args.dataset_name!='cnn_dailymail' and args.dataset_name!='trivia_qa' and args.dataset_name!='strqa' and args.dataset_name!='gsm8k' and args.mlp_l1=='No':
    #     print("Saving labels")
    #     np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.dataset_name}_{args.token}_labels_{end}.npy', labels)

if __name__ == '__main__':
    main()