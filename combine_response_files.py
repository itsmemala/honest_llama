import os
import numpy as np
import json

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
    for end in []:
        with open(f'{args.save_path}/responses/llama_7B_trivia_qa_greedy_responses_labels_train{end}.json', 'r') as read_file:
            for line in read_file:
                labels_data.append(json.loads(line))
    
    
    
    
    

if __name__ == '__main__':
    main()