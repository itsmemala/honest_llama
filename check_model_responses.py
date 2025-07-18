import os
import numpy as np
import json
from tqdm import tqdm
import argparse

verbal_unc_list = ['dont know','don\'t know','do not know','dont have','don\'t have','do not have','cannot','cant','can\'t','unable to']

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('file_name', type=str, default=None)
    parser.add_argument('num_samples', type=int, default=None)
    args = parser.parse_args()

    # model_name = 'hl_llama_7B'
    # file_name = 'trivia_qa_sampledplus_responses_train5000'
    file_path = f'/home/local/data/ms/honest_llama_data/responses/{args.model_name}_{args.file_name}.json'
    # num_samples = 1

    questions, answers = [], []
    if 'str' in args.file_name:
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            if args.num_samples==1:
                questions.append(data['full_input_text'][i])
                answers.append(data['model_completion'][i].lower())
            else:
                for j in range(args.num_samples):
                    questions.append(data['full_input_text'][i][0])
                    answers.append(data['model_completion'][i][j].lower())
    else:
        with open(file_path, 'r') as read_file:
            data = []
            for line in read_file:
                data.append(json.loads(line))
        for row in data:
            for j in range(1,args.num_samples+1,1):
                questions.append(row['prompt'])
                answers.append(row['response'+str(j)].lower())

    count_verbal_unc = 0
    for question,answer in tqdm(zip(questions,answers)):
        if any(substring in answer for substring in verbal_unc_list):
            print('Input Prompt:',question)
            print('Model Response:',answer)
            count_verbal_unc += 1
    print('# instances with verbal uncertainty:',count_verbal_unc)

if __name__ == '__main__':
    main()