import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import json
import jsonlines
import random
# from utils import get_llama_logits
import llama
import argparse

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    print('Loading model..', end=' ', flush=True)
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    print('Loading data..', end=' ', flush=True)
    # Load data
    train_data = []
    with jsonlines.open('annomi_nlg_train.jsonl') as f:
        for line in f.iter():
            train_data.append(line)
    train_reflection_indexes = []
    for i, row in enumerate(train_data):
        if 'reflection' in row['prompt']:
            train_reflection_indexes.append(i)
    
    # Prepare prompts
    train_prompts = []
    end=500
    for i in train_reflection_indexes[:end]:
        prompt = "Below is a counselling conversation between a therapist and a client. Generate the last therapist response.\n"
        prompt += train_data[i]['prompt']
        # prompt += "\n An appropriate response from the therapist to the above context would be:"
        train_prompts.append(prompt)
        # print('Prompt:',prompt)
        # break
    
    # Tokenize prompts
    tokenized_prompts = []
    for prompt in train_prompts:
        tokenized_prompts.append(tokenizer(prompt, return_tensors="pt").input_ids)
    
    print('Getting model responses..', flush=True)
    # Get model responses
    responses = []
    for i,prompt in enumerate(tqdm(tokenized_prompts)):
        prompt = prompt.to(device)
        response = model.generate(prompt, max_new_tokens=512, num_beams=1, do_sample=False, num_return_sequences=1)[:, prompt.shape[-1]:]
        # print(prompt.shape, response)
        response = tokenizer.decode(response[0], skip_special_tokens=True)
        responses.append({  'prompt':train_prompts[i]
                            ,'response1':response})
        # print('Response:',response)
    
    with open(f'{args.save_path}/responses/{args.model_name}_annomi_greedy_responses_{end}.json', 'w') as outfile:
        for entry in responses:
            json.dump(entry, outfile)
            outfile.write('\n')
    

if __name__ == '__main__':
    main()