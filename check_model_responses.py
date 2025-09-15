import os, sys
import numpy as np
import json
from tqdm import tqdm
import argparse
from collections import Counter

import torch
from sklearn.metrics import roc_auc_score
from transformers import DebertaV2Tokenizer,DebertaV2ForSequenceClassification
nli_tokenizer = DebertaV2Tokenizer.from_pretrained('khalidalt/DeBERTa-v3-large-mnli')
nli_model = DebertaV2ForSequenceClassification.from_pretrained('khalidalt/DeBERTa-v3-large-mnli').to('cuda')

import llama
from transformers import AutoTokenizer

verbal_unc_list = ['dont know','don\'t know','do not know','dont have','don\'t have','do not have','cannot','cant','can\'t','unable to']

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
    'flan_33B': 'timdettmers/qlora-flan-33b',
    'llama3.1_8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1_8B_Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma_2B': 'google/gemma-2b',
    'gemma_7B': 'google/gemma-7b'
}

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('file_name', type=str, default=None)
    parser.add_argument('labels_file_name', type=str, default=None)
    parser.add_argument('num_samples', type=int, default=None)
    args = parser.parse_args()

    # model_name = 'hl_llama_7B'
    # file_name = 'trivia_qa_sampledplus_responses_train5000'
    save_path = '/home/local/data/ms/honest_llama_data/responses'
    file_path = f'{save_path}/{args.model_name}_{args.file_name}.json'
    # num_samples = 1

    MODEL = HF_NAMES[args.model_name]
    if "llama3" in args.model_name:
        from base_transformers.models import llama3
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        llm = llama3.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    elif "gemma" in args.model_name:
        from base_transformers.models import gemma
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        llm = gemma.GemmaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        llm = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    questions, answers, maj_vote_results = [], [], []
    labels = []
    if 'str' in args.file_name:
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        with open(file_path.replace('sampled','baseline'), 'r') as read_file:
            greedy_data = json.load(read_file)
        selfcheck_scores = []
        for i in range(len(data['full_input_text'])):
            label = 0 if greedy_data['is_correct'][i]==True else 1
            labels.append(label)
            if args.num_samples==1:
                questions.append(data['full_input_text'][i])
                answers.append(data['model_completion'][i].lower())
            else:
                # marginalised_answers = []
                eval_sent = greedy_data['model_completion'][i]
                contra_scores = []
                for j in range(args.num_samples):
                    # if data['model_answer'][i][j] is None:
                    #     data['model_answer'][i][j] = False
                    questions.append(data['full_input_text'][i][0])
                    answers.append(data['model_completion'][i][j].lower())
                    # if data['model_answer'][i][j] is not None:
                    #     marginalised_answers.append(data['model_answer'][i][j])
                    #     if data['is_correct'][i][0]==True:
                    #         gt_answer = data['model_answer'][i][0]
                    #     else:
                    #         gt_answer = not data['model_answer'][i][0]
                    # for eval_sent_k in eval_sent.split(". "):
                        # inputs = nli_tokenizer.batch_encode_plus(
                        # batch_text_or_text_pairs=[(eval_sent_k, data['model_completion'][i][j])],
                        # add_special_tokens=True, padding="longest",
                        # truncation=True, return_tensors="pt",
                        # return_token_type_ids=True, return_attention_mask=True,
                        # ).to('cuda')
                        # logits = nli_model(**inputs).logits # neutral is already removed
                        # probs = torch.softmax(logits, dim=-1)
                        # prob_ = probs[0][1].item() # prob(contradiction)
                    prompt = f"Context: {data['model_completion'][i][j]}\nSentence: {eval_sent}\nIs the sentence supported by the context above?\nAnswer YES or NO.\nAnswer:"
                    tokenized_prompt = tokenizer(prompt, return_tensors = 'pt').input_ids.to('cuda')
                    llm_contra_resp = llm.generate(tokenized_prompt, max_new_tokens=5, do_sample=False, num_return_sequences=1)[:, tokenized_prompt.shape[-1]:]
                    llm_contra_resp = tokenizer.decode(llm_contra_resp[0], skip_special_tokens=True).lower()
                    prob_ = 0 if 'yes' in llm_contra_resp else 1.0 if 'no' in llm_contra_resp else 0.5
                    contra_scores.append(prob_)
                selfcheck_scores.append(np.mean(contra_scores))
                # if len(marginalised_answers)>0:
                #     maj_vote = Counter(marginalised_answers).most_common(1)[0][0]
                #     maj_vote_results.append(1 if maj_vote==gt_answer else 0)
                # # print(marginalised_answers, Counter(marginalised_answers),Counter(marginalised_answers).most_common(1)[0][0])
                # print(gt_answer,type(gt_answer),maj_vote,type(maj_vote))
                # print('\n')
                # if i==10: break
                
    else:
        with open(file_path, 'r') as read_file:
            data = []
            for line in read_file:
                data.append(json.loads(line))
        with open(file_path.replace('sampled','greedy'), 'r') as read_file:
            greedy_data = []
            for line in read_file:
                greedy_data.append(json.loads(line))
        selfcheck_scores = []
        for i,row in tqdm(enumerate(data)):
            eval_sent = greedy_data[i]['response1']
            contra_scores = []
            for j in range(1,args.num_samples+1,1):
                questions.append(row['prompt'])
                answers.append(row['response'+str(j)].lower())
                # inputs = tokenizer.batch_encode_plus(
                #     batch_text_or_text_pairs=[(eval_sent, row['response'+str(j)])],
                #     add_special_tokens=True, padding="longest",
                #     truncation=True, return_tensors="pt",
                #     return_token_type_ids=True, return_attention_mask=True,
                # ).to('cuda')
                # logits = nli_model(**inputs).logits # neutral is already removed
                # probs = torch.softmax(logits, dim=-1)
                # prob_ = probs[0][1].item() # prob(contradiction)
                prompt = f"Context: {row['response'+str(j)]}\nSentence: {eval_sent}\nIs the sentence supported by the context above?\nAnswer YES or NO.\nAnswer:"
                tokenized_prompt = tokenizer(prompt, return_tensors = 'pt').input_ids.to('cuda')
                llm_contra_resp = llm.generate(tokenized_prompt, max_new_tokens=5, do_sample=False, num_return_sequences=1)[:, tokenized_prompt.shape[-1]:]
                llm_contra_resp = tokenizer.decode(llm_contra_resp[0], skip_special_tokens=True).lower()
                # print('Model Response:',llm_contra_resp)
                # sys.exit()
                prob_ = 0 if 'yes' in llm_contra_resp else 1.0 if 'no' in llm_contra_resp else 0.5
                contra_scores.append(prob_)
            selfcheck_scores.append(np.mean(contra_scores))
        labels_file_path = f'{save_path}/{args.model_name}_{args.labels_file_name}.json'
        with open(labels_file_path, 'rb') as read_file:
            for line in read_file:
                data = json.loads(line)
                label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                labels.append(label)
        
    # print('Majority Vote Accuracy:',sum(maj_vote_results)/len(maj_vote_results),len(maj_vote_results))
    print('SelfCheck AUC',roc_auc_score(labels,selfcheck_scores))

    # count_verbal_unc = 0
    # for question,answer in tqdm(zip(questions,answers)):
    #     if any(substring in answer for substring in verbal_unc_list):
    #         print('Input Prompt:',question)
    #         print('Model Response:',answer)
    #         count_verbal_unc += 1
    # print('# instances with verbal uncertainty:',count_verbal_unc)



if __name__ == '__main__':
    main()