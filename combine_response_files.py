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
    args = parser.parse_args()
    
    # greedy_labels_data = []
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_greedy_responses_labels_train2000.json', 'r') as read_file:
    #     for line in read_file:
    #         greedy_labels_data.append(json.loads(line))
    # sampled_labels_data = []
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_sampled_responses_labels_train2000.json', 'r') as read_file:
    #     for line in read_file:
    #         sampled_labels_data.append(json.loads(line))
    # for i,g_row in enumerate(greedy_labels_data):
    #     sampled_labels_data[i]['rouge1_to_target_response11'] = g_row['rouge1_to_target']
        
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_sampledplus_responses_labels_train2000.json', 'w') as outfile:
    #     for entry in sampled_labels_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    # greedy_resp_data = []
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_greedy_responses_train2000.json', 'r') as read_file:
    #     for line in read_file:
    #         greedy_resp_data.append(json.loads(line))
    # sampled_resp_data = []
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_sampled_responses_train2000.json', 'r') as read_file:
    #     for line in read_file:
    #         sampled_resp_data.append(json.loads(line))
    # for i,g_row in enumerate(greedy_resp_data):
    #     sampled_resp_data[i]['response11'] = g_row['response1']
        
    # with open(f'{args.save_path}/responses/llama_2_7B_trivia_qa_sampledplus_responses_train2000.json', 'w') as outfile:
    #     for entry in sampled_resp_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')


    # labels_data = []
    # for end in [10000,20000]:
    #     with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_greedy_responses_labels_train{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             labels_data.append(json.loads(line))
    # train_len = 20000
    # print(train_len)
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_greedy_responses_labels_train20000.json', 'w') as outfile:
    #     for entry in labels_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # # with open(f'{args.save_path}/responses/hl_llama_7B_trivia_qa_greedy_responses_labels_test.json', 'w') as outfile:
    # #     for entry in labels_data[train_len:]:
    # #         json.dump(entry, outfile)
    # #         outfile.write('\n')
    
    # response_data = []
    # for end in [10000,20000]:
    #     with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_greedy_responses_train{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             response_data.append(json.loads(line))
    # train_len = 20000
    # num_correct = 0
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_greedy_responses_train20000.json', 'w') as outfile:
    #     for entry in response_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'w') as outfile:
    # #     for entry in response_data[train_len:]:
    # #         json.dump(entry, outfile)
    # #         outfile.write('\n')


    ## strqa
    # response_data = []
    # for end in ['']:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses{end}.json', 'r') as read_file:
    #         response_data = json.load(read_file)
    # response_data_pd = pd.DataFrame.from_dict(response_data)
    # print(response_data_pd.columns)
    # response_data_pd['index'] = response_data_pd.index
    # print(response_data_pd.columns)
    # train, test = train_test_split(response_data_pd, test_size=0.2, stratify=response_data_pd['is_correct'])
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_train.json', 'w') as outfile:
    #     json.dump(train.to_dict(orient='list'), outfile)
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'w') as outfile:
    #     json.dump(test.to_dict(orient='list'), outfile)
    
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_dola_0to16_responses.json', 'r') as read_file:
    #     mitigated_response_data = json.load(read_file)
    # mitigated_response_data_pd = pd.DataFrame.from_dict(mitigated_response_data)
    # mitigated_response_data_pd = mitigated_response_data_pd.iloc[test['index'],:]
    # print(len(test),len(mitigated_response_data_pd))
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_dola_0to16_responses_test.json', 'w') as outfile:
    #     json.dump(mitigated_response_data_pd.to_dict(orient='list'), outfile)

    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_train.json', 'r') as read_file:
    #     response_data = json.load(read_file)
    # print(len(response_data['is_correct']))

    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'r') as read_file:
    #     response_data = json.load(read_file)
    # print(sum(response_data['is_correct'])/len(response_data['is_correct']))
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_dola_0to16_responses_test.json', 'r') as read_file:
    #     response_data = json.load(read_file)
    # print(sum(response_data['is_correct'])/len(response_data['is_correct']))

    with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_train.json', 'r') as read_file:
        train_data = json.load(read_file)
    train_data_pd = pd.DataFrame.from_dict(train_data)
    print(len(train_data_pd))
    with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'r') as read_file:
        test_data = json.load(read_file)
    test_data_pd = pd.DataFrame.from_dict(test_data)
    print(len(test_data_pd))
    print(train_data_pd[:2])
    with open(f'{args.save_path}/responses/hl_llama_7B_strqa_sampled_responses_validation0.json', 'r') as read_file:
        response_data = json.load(read_file)
    response_data_pd = pd.DataFrame.from_dict(response_data)
    response_data_pd['index'] = response_data_pd.index
    train = response_data_pd.loc[response_data_pd['index'].isin(train_data_pd['index'].tolist())]
    test = response_data_pd.loc[response_data_pd['index'].isin(test_data_pd['index'].tolist())]
    print(train[:2])
    print(len(train_data_pd['index'].tolist()),len(test_data_pd['index'].tolist()),len(response_data_pd['index']))
    print(len(train),len(test))
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_sampled_responses_train.json', 'w') as outfile:
    #     json.dump(train.to_dict(orient='list'), outfile)
    # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_sampled_responses_test.json', 'w') as outfile:
    #     json.dump(test.to_dict(orient='list'), outfile)
    ##

    ## tqa_gen
    # dataset = load_dataset("truthful_qa", "generation", streaming= True)['validation']
    # len_dataset = 817
    # all_indexes = np.arange(len_dataset)
    # train_idxs = np.random.choice(all_indexes, size=int(len_dataset*(1-0.2)), replace=False)
    # test_idxs = np.array([x for x in all_indexes if x not in train_idxs])
    # train_data, train_labels, test_data, test_labels = [], [], [], []
    # train_cor_num, test_cor_num = 0, 0
    # for idx,val in enumerate(list(dataset.take(len_dataset))):
    #     for ans in val['correct_answers']:
    #         if idx in train_idxs:
    #             train_data.append({'prompt':val['question'],'response1':ans})
    #             train_labels.append({'rouge1_to_target':1})
    #             train_cor_num += 1
    #         else:
    #             test_data.append({'prompt':val['question'],'response1':ans})
    #             test_labels.append({'rouge1_to_target':1})
    #             test_cor_num += 1
    #     for ans in val['incorrect_answers']:
    #         if idx in train_idxs:
    #             train_data.append({'prompt':val['question'],'response1':ans})
    #             train_labels.append({'rouge1_to_target':0})
    #         else:
    #             test_data.append({'prompt':val['question'],'response1':ans})
    #             test_labels.append({'rouge1_to_target':0})
    # print(len(train_data),len(test_data))
    # print(train_cor_num/len(train_labels),test_cor_num/len(test_labels))
    # with open(f'{args.save_path}/responses/tqa_gen_greedy_responses_train.json', 'w') as outfile:
    #     for entry in train_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/tqa_gen_greedy_responses_test.json', 'w') as outfile:
    #     for entry in test_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/tqa_gen_greedy_responses_labels_train.json', 'w') as outfile:
    #     for entry in train_labels:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/tqa_gen_greedy_responses_labels_test.json', 'w') as outfile:
    #     for entry in test_labels:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

if __name__ == '__main__':
    main()