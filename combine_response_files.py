import os
import numpy as np
import json
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
    
    labels_data = []
    for end in [500,1500,3000,6800]:
        with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_labels_train{end}.json', 'r') as read_file:
            for line in read_file:
                labels_data.append(json.loads(line))
    with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_labels_train5000.json', 'w') as outfile:
        for entry in labels_data[:5000]:
            json.dump(entry, outfile)
            outfile.write('\n')
    with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_labels_train1800.json', 'w') as outfile:
        for entry in labels_data[5000:1800]:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    response_data = []
    for end in [500,1500,3000,6800]:
        with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_train{end}.json', 'r') as read_file:
            for line in read_file:
                labels_data.append(json.loads(line))
    with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_train5000.json', 'w') as outfile:
        for entry in response_data[:5000]:
            json.dump(entry, outfile)
            outfile.write('\n')
    with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_train1800.json', 'w') as outfile:
        for entry in response_data[5000:1800]:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    main()