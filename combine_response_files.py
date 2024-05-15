import os
import numpy as np
import json
import pandas as pd
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
    args = parser.parse_args()
    
    # labels_data = []
    # for end in ['']:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_baseline_responses{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             labels_data.append(json.loads(line))
    # train_len = 0.8*len(labels_data)
    # print(train_len)
    # with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_baseline_responses_labels_train.json', 'w') as outfile:
    #     for entry in labels_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_baseline_responses_labels_test.json', 'w') as outfile:
    #     for entry in labels_data[train_len:]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    
    # response_data = []
    # for end in ['']:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             response_data.append(json.loads(line))
    # train_len = 0.8*len(labels_data)
    # num_correct = 0
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_train.json', 'w') as outfile:
    #     for entry in response_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'w') as outfile:
    #     for entry in response_data[train_len:]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    response_data = []
    for end in ['']:
        with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses{end}.json', 'r') as read_file:
            response_data = json.load(read_file)
    response_data_pd = pd.DataFrame.from_dict(response_data)
    train, test = train_test_split(response_data_pd, test_size=0.2, stratify=response_data_pd['is_correct'])
    with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_train.json', 'w') as outfile:
        json.dump(train.to_dict(orient='list'), outfile)
    with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'w') as outfile:
        json.dump(test.to_dict(orient='list'), outfile)

if __name__ == '__main__':
    main()