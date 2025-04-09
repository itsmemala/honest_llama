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

      
    greedy_resp_data = []
    with open(f'{args.save_path}/responses/{args.model}_nq_open_greedy_responses_train5000.json', 'r') as read_file:
        for line in read_file:
            greedy_resp_data.append(json.loads(line))
    greedy_resp_data = greedy_resp_data#[:2000]
    sampled_resp_data = []
    with open(f'{args.save_path}/responses/{args.model}_nq_open_sampled_responses_train5000.json', 'r') as read_file:
        for line in read_file:
            sampled_resp_data.append(json.loads(line))
    for i,s_row in enumerate(sampled_resp_data):
        greedy_i = ''
        for k,g_row in enumerate(greedy_resp_data):
            if g_row['prompt']==s_row['prompt']: greedy_i = k
        if greedy_i=='': print('cant find prompt at row',i, 'prompt:',s_row['prompt'])
        try:
            sampled_resp_data[i]['response11'] = greedy_resp_data[greedy_i]['response1']
        except TypeError:
            print(greedy_i)
            sampled_resp_data[i]['response11'] = ''

    print(len(sampled_resp_data))    
    with open(f'{args.save_path}/responses/{args.model}_nq_open_sampledplus_responses_train5000.json', 'w') as outfile:
        for entry in sampled_resp_data:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    greedy_labels_data = []
    with open(f'{args.save_path}/responses/{args.model}_nq_open_greedy_responses_labels_train5000.json', 'r') as read_file:
        for line in read_file:
            greedy_labels_data.append(json.loads(line))
    greedy_labels_data = greedy_labels_data#[:2000]
    sampled_labels_data = []
    # for end in [1000,2000,3000,4000,5000]:
    #     with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampled_responses_labels_train{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             sampled_labels_data.append(json.loads(line))
    with open(f'{args.save_path}/responses/{args.model}_nq_open_sampled_responses_labels_train5000.json', 'r') as read_file:
        for line in read_file:
            sampled_labels_data.append(json.loads(line))
    for i,s_row in enumerate(sampled_resp_data):
        greedy_i = ''
        for k,g_row in enumerate(greedy_resp_data):
            if g_row['prompt']==s_row['prompt']: greedy_i = k
        try:
            sampled_labels_data[i]['rouge1_to_target_response11'] = greedy_labels_data[greedy_i]['rouge1_to_target']
        except TypeError:
            sampled_labels_data[i]['rouge1_to_target_response11'] = 0
        
    print(len(sampled_labels_data))
    with open(f'{args.save_path}/responses/{args.model}_nq_open_sampledplus_responses_labels_train5000.json', 'w') as outfile:
        for entry in sampled_labels_data:
            json.dump(entry, outfile)
            outfile.write('\n')


    

if __name__ == '__main__':
    main()