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
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

def clean_response(response):
    response = response.split('|')[0] # Ignore any subsequent client/therapist utterances generated
    return response

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('--prompt_type', type=str, default='A')
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--num_ret_seq', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    print('Loading model..', end=' ', flush=True)
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
        model = llama.LlamaForCausalLM.from_pretrained(
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
        model = PeftModel.from_pretrained(model, adapter_path, cache_dir=args.save_path+"/"+args.model_cache_dir)
    else:
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
    for idx,i in enumerate(train_reflection_indexes[:end]):
        if args.prompt_type=='A':
            prompt = "Below is a counselling conversation between a therapist and a client. Generate the last therapist response.\n" # prompt-A (llama7B greedy)
            prompt += train_data[i]['prompt']
        elif args.prompt_type=='B':
            prompt = "Below is a counselling conversation between a therapist and a client.\n"
            prompt += train_data[i]['prompt']
            prompt += "\n An appropriate response from the therapist to the above context would be:" #prompt-B (flan33B greedy)
        if args.prompt_type=='C':
            prompt = "Below is a counselling conversation between a therapist and a client. Generate the last therapist response.\n"
            prompt += train_data[i]['prompt'].replace('~<reflection>','')
        elif args.prompt_type=='D':
            prompt = "Below is a counselling conversation between a therapist and a client.\n"
            prompt += train_data[i]['prompt'].replace('<therapist>~<reflection>','')
            prompt += "\n An appropriate response from the therapist to the above context would be:\n<therapist>"
        train_prompts.append(prompt)
        # if idx==0: print('Prompt:',prompt)
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
        if args.model_name=='flan_33B':
            response = model.generate(input_ids=prompt,
                                    generation_config=GenerationConfig(
                                        max_new_tokens=512
                                        # ,num_beams=1
                                        ,top_p=args.top_p
                                        ,do_sample=args.do_sample
                                        ,num_return_sequences=args.num_ret_seq
                                    )
                                )[:, prompt.shape[-1]:]
        else:
            response = model.generate(prompt, max_new_tokens=512, 
                                        # num_beams=1,
                                        top_p=args.top_p, do_sample=args.do_sample, num_return_sequences=args.num_ret_seq)[:, prompt.shape[-1]:]
        # print(prompt.shape, response)
        if args.num_ret_seq==1:
            response = tokenizer.decode(response[0], skip_special_tokens=True)
            response = clean_response(response)
            responses.append({  'prompt':train_prompts[i]
                                ,'response1':response})
            print(i,'Response:',response,'\n')
        else:
            resp_dict = {'prompt':train_prompts[i]}
            for j in range(args.num_ret_seq):
                cur_response = tokenizer.decode(response[j], skip_special_tokens=True)
                cur_response = clean_response(cur_response)
                resp_dict['response'+str(j+1)] = cur_response
                print(i,j,'Response:',cur_response,'\n')
            responses.append(resp_dict)
    
    file_name = f'{args.save_path}/responses/{args.model_name}_annomi_{args.prompt_type}_greedy_responses_{end}.json' if args.do_sample==False else f'{args.save_path}/responses/{args.model_name}_annomi_{args.prompt_type}_sampled_responses_{end}.json'
    with open(file_name, 'w') as outfile:
        for entry in responses:
            json.dump(entry, outfile)
            outfile.write('\n')
    

if __name__ == '__main__':
    main()