import os
import numpy as np
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(): 
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--model',type=str, default='')
    args = parser.parse_args()

    ##gsm8k
    with open(f'{args.save_path}/responses/{args.model}_gsm8k_greedy_responses_train5000.json', 'r') as read_file:
        greedy_train_data = json.load(read_file)
    with open(f'{args.save_path}/responses/{args.model}_gsm8k_sampled_responses_train2000.json', 'r') as read_file:
        sampled_train_data = json.load(read_file)
    # print(len(sampled_train_data['is_correct']))
    # print(sampled_train_data.keys())
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    for i in range(len(sampled_train_data['is_correct'])):
        if len(sampled_train_data['is_correct'][i])>0:
            # print(sampled_train_data['is_correct'][i], [greedy_train_data['is_correct'][i]])
            greedy_i = ''
            for j in range(len(greedy_train_data['full_input_text'])):
                if greedy_train_data['full_input_text'][j]==sampled_train_data['full_input_text'][i][0]: greedy_i=j # needed to match rows when generated using parallel gpus
            if greedy_i == '': print(i,sampled_train_data['full_input_text'][i])
            result_dict['is_correct'].append(sampled_train_data['is_correct'][i] + [greedy_train_data['is_correct'][greedy_i]])
            result_dict['model_answer'].append(sampled_train_data['model_answer'][i] + [greedy_train_data['model_answer'][greedy_i]])
            result_dict['model_completion'].append(sampled_train_data['model_completion'][i] + [greedy_train_data['model_completion'][greedy_i]])
            result_dict['full_input_text'].append(sampled_train_data['full_input_text'][i] + [greedy_train_data['full_input_text'][greedy_i]])
    
    print(len(result_dict['is_correct'])) # 1999
    with open(f'{args.save_path}/responses/{args.model}_gsm8k_sampledplus_responses_train2000.json', 'w') as f:
            json.dump(result_dict, f)

    
if __name__ == '__main__':
    main()