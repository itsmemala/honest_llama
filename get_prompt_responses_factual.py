import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import numpy as np
import pickle
import string
import re
import json
import jsonlines
import random
import llama
import argparse
# from transformers import BitsAndBytesConfig, GenerationConfig
# from peft import PeftModel
# from peft.tuners.lora import LoraLayer
import evaluate

# Squad F1 calculation from: https://github.com/tangbinh/question-answering/blob/master/evaluate.py

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def my_squad_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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
    parser.add_argument('dataset_name', type=str, default='nq_open', help='dataset for querying the model')
    parser.add_argument('--len_dataset', type=int, default=0)
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--use_split', type=str, default='validation')
    parser.add_argument('--hallu_check_prompt', type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--num_ret_seq', type=int, default=1)
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
    # if args.num_ret_seq>1 and args.model_name=='alpaca_7B': os.environ["PYTORCH_USE_CUDA_DSA"] = "1" #tokenizer.pad_token = tokenizer.eos_token
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    if args.num_ret_seq>1 and args.model_name=='llama_2_7B': model = model.bfloat16() # Numerical instability; Solution from: https://github.com/meta-llama/llama/issues/380
    device = "cuda"
    # device = 'cpu' # for debugging
    # model = model.cpu()

    print('Loading data..')
    print(args.len_dataset,args.start_at,args.use_split)
    # Load data
    len_dataset = args.len_dataset
    start_at = args.start_at
    if args.dataset_name=='nq_open':
        hf_dataset_name = 'nq_open'
        dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
    elif args.dataset_name=='trivia_qa':
        hf_dataset_name = 'mandarjoshi/trivia_qa'
        dataset = load_dataset(hf_dataset_name, 'rc.nocontext', streaming= True)[args.use_split]
    elif args.dataset_name=='cnn_dailymail':
        hf_dataset_name = 'cnn_dailymail'
        dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
    if args.hallu_check_prompt is None:
        prompts = []
        tokenized_prompts = []
        for idx,val in enumerate(list(dataset.take(len_dataset))[start_at:]):
            if args.dataset_name=='nq_open':
                question = val['question']
                cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
            elif args.dataset_name=='trivia_qa':
                question = val['question']
                cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
            elif args.dataset_name=='cnn_dailymail':
                article = val['article']
                cur_prompt = f"Article: {article}\n Summarize the article in two to three sentences. Summary: "
            prompts.append(cur_prompt)
            tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    else:
        # Load greedy responses
        greedy_resp_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_greedy_responses_{args.use_split}{args.len_dataset}.json'
        with open(greedy_resp_fname, 'r') as read_file:
            greedy_resp_data = []
            for line in read_file:
                greedy_resp_data.append(json.loads(line))
        prompts = []
        tokenized_prompts = []
        for row,val in zip(greedy_resp_data,list(dataset.take(len_dataset))[start_at:]):
            if args.hallu_check_prompt==1:
                cur_prompt = row['prompt'] + row['response1'] + "\n The above generated answer is incorrect. Revised answer: "
            if args.hallu_check_prompt==2:
                cur_prompt = row['prompt'] + row['response1'] + "\n The above answer may be incorrect. The actual correct answer is: "
            if args.hallu_check_prompt==3 and (args.dataset_name=='trivia_qa' or args.dataset_name=='nq_open'):
                question = val['question']
                cur_prompt = f"This is a bot that correctly answers questions. Consider the below question and a possible answer, which may or may not be correct. Provide the correct answer to the question. \n Q: {question} Possible answer: {row['response1']}\n Correct answer:"
            prompts.append(cur_prompt)
            tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    
    print('Getting model responses..')
    # Get model responses
    responses = []
    if args.dataset_name=='nq_open' or args.dataset_name=='trivia_qa':
        # period_token_id = tokenizer('. ')['input_ids'][1]
        period_token_id = tokenizer('.')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\"]
        # question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
        # eos_tokens = ['Question', '\n', 'Answer', ':']
        checkgens = ['QA2:','Q.', 'B:']
    elif args.dataset_name=='cnn_dailymail':
        period_token_id = tokenizer('\n')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\", 'Summary:', ' Summary:']
        checkgens = ['Summary:']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in eos_tokens]
    # print('Bad word ids:',question_framing_ids)
    for i,tokenized_prompt in enumerate(tqdm(tokenized_prompts)):
        tokenized_prompt = tokenized_prompt.to(device)
        response = model.generate(tokenized_prompt, max_new_tokens=512,
                                    # num_beams=1,
                                    temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample, num_return_sequences=args.num_ret_seq,
                                    eos_token_id=period_token_id,
                                    bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
                                    )[:, tokenized_prompt.shape[-1]:]
        if args.num_ret_seq==1:
            response = tokenizer.decode(response[0], skip_special_tokens=True)
            for check_gen in checkgens: # Fix generation stopping errors
                # before_trunc = response
                response = response.split(check_gen)[0]
                # if before_trunc=="":
                #     print(i)
            responses.append({'prompt':prompts[i],
                                'response1':response})
        else:
            resp_dict = {'prompt':prompts[i]}
            for j in range(args.num_ret_seq):
                cur_response = tokenizer.decode(response[j], skip_special_tokens=True)
                for check_gen in checkgens: # Fix generation stopping errors
                    cur_response = cur_response.split(check_gen)[0]
                resp_dict['response'+str(j+1)] = cur_response
                # print(i,j,'Response:',cur_response,'\n')
            responses.append(resp_dict)
    
    # batches = [(0,10)]
    # for batch_start,batch_end in batches:
    #     tokenized_prompt = tokenizer(prompts[batch_start:batch_end], return_tensors = 'pt').input_ids
    #     tokenized_prompt = tokenized_prompt.to(device)
    #     response = model.generate(tokenized_prompt, max_new_tokens=512, num_beams=1, do_sample=False, num_return_sequences=1,
    #                                 eos_token_id=period_token_id,
    #                                 bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
    #                                 )[:, tokenized_prompt.shape[-1]:]
    #     response = tokenizer.decode(response, skip_special_tokens=True)
    #     for i,resp in enumerate(response):
    #         for check_gen in checkgens: # Fix generation stopping errors
    #             resp = resp.split(check_gen)[0]
    #         responses.append({'prompt':prompts[batch_start+i],
    #                         'response1':resp})
    
    print('Saving model responses..')
    if args.hallu_check_prompt is None:
        gen_type = 'sampled' if args.do_sample else 'greedy'
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}.json'
    else:
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_{args.use_split}{args.len_dataset}.json'
    with open(save_fname, 'w') as outfile:
        for entry in responses:
            json.dump(entry, outfile)
            outfile.write('\n')

    # gen_type = 'sampled' if args.do_sample else 'greedy'
    # resp_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}.json'
    # # resp_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_{args.use_split}{args.len_dataset}.json'
    # with open(resp_fname, 'r') as read_file:
    #     responses = []
    #     for line in read_file:
    #         responses.append(json.loads(line))
    
    # for i,row in enumerate(responses):
    #     for j in range(args.num_ret_seq):
    #         resp = row['response'+str(j+1)]
    #         responses[i]['response'+str(j+1)] = resp.split("\n")[0]
    #     # if resp.split("\n")[0]!=resp:
    #     #     print(i,"\n",resp,"\n\n",resp.split("\n")[0])
    # print('Saving model responses..')
    # save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}.json'
    # with open(save_fname, 'w') as outfile:
    #     for entry in responses:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    
    print('Getting labels for model responses..')
    labels = []
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    squad_metrics = evaluate.load('squad')
    for i,batch in tqdm(enumerate(list(dataset.take(args.len_dataset))[start_at:])): # one row at a time
        if args.num_ret_seq==1:
            labels_dict = {'exact_match': 0.0,
                            'rouge1_to_target':0.0,
                            'rouge2_to_target':0.0,
                            'rougeL_to_target':0.0,
                            'squad_f1':0.0}
        else:
            labels_dict = {}
            for j in range(args.num_ret_seq):
                labels_dict['exact_match_response'+str(j+1)]=0.0
                labels_dict['rouge1_to_target_response'+str(j+1)]=0.0
                labels_dict['rouge2_to_target_response'+str(j+1)]=0.0
                labels_dict['rougeL_to_target_response'+str(j+1)]=0.0
                labels_dict['squad_f1_response'+str(j+1)]=0.0
        if args.dataset_name=='nq_open':
            reference_answers = batch['answer'] 
        elif args.dataset_name=='trivia_qa':
            reference_answers_unformatted = batch['answer']
            reference_answers = reference_answers_unformatted['aliases'] + reference_answers_unformatted['normalized_aliases']
        elif args.dataset_name=='cnn_dailymail':
            reference_answers = [batch['highlights']]
        for answer in reference_answers:
            for j in range(args.num_ret_seq):
                resp_wise_label_name = '_response'+str(j+1) if args.num_ret_seq>1 else ''
                # predictions, predictions_dict = [responses[j]['response1'].lstrip()], [{'prediction_text':responses[j]['response1'].lstrip()}]
                # references, references_dict = [answer], [{'answers':{'text':[answer]}}]
                predictions = [responses[i]['response'+str(j+1)].lstrip()]
                references = [answer]
                results = exact_match_metric.compute(predictions=predictions,
                                                        references=references,
                                                        ignore_case=True,
                                                        ignore_punctuation=True)
                labels_dict['exact_match' + resp_wise_label_name] = max(results['exact_match'], labels_dict['exact_match' + resp_wise_label_name])
                rouge_results = rouge.compute(predictions=predictions, references=references)
                for rouge_type in ['rouge1','rouge2','rougeL']:
                    labels_dict[rouge_type + '_to_target' + resp_wise_label_name] = max(rouge_results[rouge_type],
                                                                    labels_dict[rouge_type + '_to_target' + resp_wise_label_name])
                squad_f1 = my_squad_f1_score(predictions[0],references[0])
                labels_dict['squad_f1' + resp_wise_label_name] = max(squad_f1, labels_dict['squad_f1' + resp_wise_label_name])

        labels.append(labels_dict)


    print('Saving labels..')
    if args.hallu_check_prompt is None:
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_labels_{args.use_split}{args.len_dataset}.json'
    else:
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_labels_{args.use_split}{args.len_dataset}.json'
    with open(save_fname, 'w') as outfile:
        for entry in labels:
            json.dump(entry, outfile)
            outfile.write('\n')
    

if __name__ == '__main__':
    main()