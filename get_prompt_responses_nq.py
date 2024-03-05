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
from utils import tokenized_nq_orig
import llama
import argparse
# from transformers import BitsAndBytesConfig, GenerationConfig
# from peft import PeftModel
# from peft.tuners.lora import LoraLayer
import evaluate

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
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    print('Loading model..')
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    print('Loading data..')
    # Load data
    len_dataset = 50 # 3610
    dataset = load_dataset("nq_open", streaming= True)['validation']
    prompts = []
    tokenized_prompts = []
    for val in list(dataset.take(len_dataset)):
        question = val['question']
        cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
        prompts.append(cur_prompt)
        tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
        tokenized_prompts.append(tokenized_prompt)
    
    print('Getting model responses..')
    # Get model responses
    responses = []
    # period_token_id = tokenizer('. ')['input_ids'][1]
    period_token_id = tokenizer('.')['input_ids']
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                    'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\"]
    # question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
    # eos_tokens = ['Question', '\n', 'Answer', ':']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in eos_tokens]
    print('Bad word ids:',question_framing_ids)
    for i,prompt in enumerate(tqdm(tokenized_prompts)):
        prompt = prompt.to(device)
        response = model.generate(prompt, max_new_tokens=256, num_beams=1, do_sample=False, num_return_sequences=1,
                                    eos_token_id=period_token_id,
                                    bad_words_ids=question_framing_ids + [prompt.tolist()[0]]
                                    )[:, prompt.shape[-1]:]
        response = tokenizer.decode(response[0], skip_special_tokens=True)
        for check_gen in ['QA2:','Q.', 'B:']
            response = response.split(check_gen)[0]
        responses.append({'prompt':prompts[i],
                            'response1':response})
    
    print('Saving model responses..')
    with open(f'{args.save_path}/responses/{args.model_name}_nq_greedy_responses.json', 'w') as outfile:
        for entry in responses:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    print('Getting labels for model responses..')
    labels = []
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    for i,batch in enumerate(list(dataset.take(len_dataset))): # one row at a time
        labels_dict = {'exact_match': 0.0,
                        'rouge1_to_target':0.0,
                        'rouge2_to_target':0.0,
                        'rougeL_to_target':0.0}
        reference_answers = batch['answer']
        for answer in reference_answers:
            predictions = [responses[i]['response1'].lstrip()]
            references = [answer]
            results = exact_match_metric.compute(predictions=predictions,
                                                    references=references,
                                                    ignore_case=True,
                                                    ignore_punctuation=True)
            labels_dict['exact_match'] = max(results['exact_match'], labels_dict['exact_match'])
            rouge_results = rouge.compute(predictions=predictions, references=references)
            for rouge_type in ['rouge1','rouge2','rougeL']:
                labels_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                                labels_dict[rouge_type + '_to_target'])

        labels.append(labels_dict)


    print('Saving labels..')
    with open(f'{args.save_path}/responses/{args.model_name}_nq_greedy_responses_labels.json', 'w') as outfile:
        for entry in labels:
            json.dump(entry, outfile)
            outfile.write('\n')
    

if __name__ == '__main__':
    main()