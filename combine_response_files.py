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

    ### update labels ###
    # sampled_labels_data = []
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampledplus_responses_labels_train5000.json', 'r') as read_file:
    #     for line in read_file:
    #         sampled_labels_data.append(json.loads(line))
    #         # print(json.loads(line))
    #         # break
    # for i,s_row in enumerate(sampled_labels_data):
    #     for j in range(1,11,1):
    #         sampled_labels_data[i]['rouge1_to_target_response'+str(j)] = sampled_labels_data[i]['rouge1_to_target_response11']

    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampledplussl_responses_labels_train5000.json', 'w') as outfile:
    #     for entry in sampled_labels_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    # strqa, gsm8k
    # with open(f'{args.save_path}/responses/gemma_2B_gsm8k_sampledplus_responses_train5000.json', 'r') as read_file:
    #     sampled_train_data = json.load(read_file)
    # for i in range(len(sampled_train_data['is_correct'])):
    #     # print(sampled_train_data['is_correct'][i])
    #     temp_labels_list = [sampled_train_data['is_correct'][i][-1] for j in range(len(sampled_train_data['is_correct'][i]))]
    #     # print(temp_labels_list)
    #     # break
    #     sampled_train_data['is_correct'][i] = temp_labels_list
    
    # with open(f'{args.save_path}/responses/gemma_2B_gsm8k_sampledplussl_responses_train5000.json', 'w') as f:
    #     json.dump(sampled_train_data, f)

    ###


    # num_samples = 11
    # prompt_wise_labels = []
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_sampledplus_responses_labels_train2000.json', 'r') as read_file:
    #     for line in read_file:
    #         data = json.loads(line)
    #         labels = []
    #         for j in range(1,num_samples+1,1):
    #             label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1
    #             labels.append(label)
    #         prompt_wise_labels.append(labels)
    
    # homo_prompts, hetero_prompts = [], []
    # all_hallu_prompts, all_nh_prompts = [], []
    # hetero_prompts_sum = []
    # for i,labels in enumerate(prompt_wise_labels):
    #     if sum(labels)==0 or sum(labels)==len(labels):
    #         homo_prompts.append(i)
    #         if sum(labels)==len(labels): all_hallu_prompts.append(i)
    #         if sum(labels)==0: all_nh_prompts.append(i)
    #     else:
    #         hetero_prompts.append(i)
    #         hetero_prompts_sum.append(sum(labels))
    # print(len(homo_prompts),len(hetero_prompts))
    # print(len(all_hallu_prompts),len(all_nh_prompts))
    # print(np.histogram(hetero_prompts_sum, bins=num_samples-1))

    # hetero_labels_data = []
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_sampledplus_responses_labels_train2000.json', 'r') as read_file:
    #     i = 0
    #     for line in read_file:
    #         if i in hetero_prompts:
    #             hetero_labels_data.append(json.loads(line))
    #         i += 1
    # hetero_resp_data = []
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_sampledplus_responses_train2000.json', 'r') as read_file:
    #     i = 0
    #     for line in read_file:
    #         if i in hetero_prompts:
    #             hetero_resp_data.append(json.loads(line))
    #         i += 1
    
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_sampledhetero_responses_labels_train2000.json', 'w') as outfile:
    #     for entry in hetero_labels_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # with open(f'{args.save_path}/responses/alpaca_7B_trivia_qa_sampledhetero_responses_train2000.json', 'w') as outfile:
    #     for entry in hetero_resp_data:
    #         json.dump(entry, outfile)
            # outfile.write('\n')
    
    # greedy_resp_data = []
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_greedy_responses_train5000.json', 'r') as read_file:
    #     for line in read_file:
    #         greedy_resp_data.append(json.loads(line))
    # greedy_resp_data = greedy_resp_data#[:2000]
    # sampled_resp_data = []
    # # for end in [1000,2000,3000,4000,5000]:
    # #     with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampled_responses_train{end}.json', 'r') as read_file:
    # #         for line in read_file:
    # #             sampled_resp_data.append(json.loads(line))
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampled_responses_train5000.json', 'r') as read_file:
    #     for line in read_file:
    #         sampled_resp_data.append(json.loads(line))
    # for i,s_row in enumerate(sampled_resp_data):
    #     greedy_i = ''
    #     for k,g_row in enumerate(greedy_resp_data):
    #         if g_row['prompt']==s_row['prompt']: greedy_i = k
    #     sampled_resp_data[i]['response11'] = greedy_resp_data[greedy_i]['response1']

    # print(len(sampled_resp_data))    
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampledplus_responses_train5000.json', 'w') as outfile:
    #     for entry in sampled_resp_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    
    # greedy_labels_data = []
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_greedy_responses_labels_train5000.json', 'r') as read_file:
    #     for line in read_file:
    #         greedy_labels_data.append(json.loads(line))
    # greedy_labels_data = greedy_labels_data#[:2000]
    # sampled_labels_data = []
    # # for end in [1000,2000,3000,4000,5000]:
    # #     with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampled_responses_labels_train{end}.json', 'r') as read_file:
    # #         for line in read_file:
    # #             sampled_labels_data.append(json.loads(line))
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampled_responses_labels_train5000.json', 'r') as read_file:
    #     for line in read_file:
    #         sampled_labels_data.append(json.loads(line))
    # for i,s_row in enumerate(sampled_resp_data):
    #     greedy_i = ''
    #     for k,g_row in enumerate(greedy_resp_data):
    #         if g_row['prompt']==s_row['prompt']: greedy_i = k
    #     sampled_labels_data[i]['rouge1_to_target_response11'] = greedy_labels_data[greedy_i]['rouge1_to_target']
        
    # print(len(sampled_labels_data))
    # with open(f'{args.save_path}/responses/gemma_2B_nq_open_sampledplus_responses_labels_train5000.json', 'w') as outfile:
    #     for entry in sampled_labels_data:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')


    # labels_data = []
    # for end in [2000,5000]:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_nq_open_greedy_responses_labels_train{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             labels_data.append(json.loads(line))
    # train_len = 5000
    # print(train_len)
    # with open(f'{args.save_path}/responses/hl_llama_7B_nq_open_greedy_responses_labels_train5000.json', 'w') as outfile:
    #     for entry in labels_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # # with open(f'{args.save_path}/responses/hl_llama_7B_trivia_qa_greedy_responses_labels_test.json', 'w') as outfile:
    # #     for entry in labels_data[train_len:]:
    # #         json.dump(entry, outfile)
    # #         outfile.write('\n')
    
    # response_data = []
    # for end in [2000,5000]:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_nq_open_greedy_responses_train{end}.json', 'r') as read_file:
    #         for line in read_file:
    #             response_data.append(json.loads(line))
    # train_len = 5000
    # num_correct = 0
    # with open(f'{args.save_path}/responses/hl_llama_7B_nq_open_greedy_responses_train5000.json', 'w') as outfile:
    #     for entry in response_data[:train_len]:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    # # with open(f'{args.save_path}/responses/hl_llama_7B_strqa_baseline_responses_test.json', 'w') as outfile:
    # #     for entry in response_data[train_len:]:
    # #         json.dump(entry, outfile)
    # #         outfile.write('\n')


    # strqa
    # response_data = []
    # for end in ['']:
    #     with open(f'{args.save_path}/responses/gemma_2B_strqa_greedy_responses_train5000.json', 'r') as read_file:
    #         response_data = json.load(read_file)
    # response_data_pd = pd.DataFrame.from_dict(response_data)
    # print(response_data_pd.columns)
    # response_data_pd['index'] = response_data_pd.index
    # print(response_data_pd.columns)
    # train, test = train_test_split(response_data_pd, test_size=0.2, stratify=response_data_pd['is_correct'])
    # with open(f'{args.save_path}/responses/gemma_2B_strqa_baseline_responses_train.json', 'w') as outfile:
    #     json.dump(train.to_dict(orient='list'), outfile)
    # with open(f'{args.save_path}/responses/gemma_2B_strqa_baseline_responses_test.json', 'w') as outfile:
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

    with open(f'{args.save_path}/responses/alpaca_7B_strqa_baseline_responses_train.json', 'r') as read_file:
        train_data = json.load(read_file)
    train_data_pd = pd.DataFrame.from_dict(train_data)
    # print(len(train_data_pd))
    with open(f'{args.save_path}/responses/alpaca_7B_strqa_baseline_responses_test.json', 'r') as read_file:
        test_data = json.load(read_file)
    test_data_pd = pd.DataFrame.from_dict(test_data)
    # print(len(test_data_pd))
    # print(train_data_pd[:2])
    with open(f'{args.save_path}/responses/alpaca_7B_strqa_sampled_responses_train5000.json', 'r') as read_file:
        response_data = json.load(read_file)
    response_data_pd = pd.DataFrame.from_dict(response_data)
    response_data_pd['index'] = response_data_pd.index
    # print(len(response_data_pd))
    train = response_data_pd.loc[response_data_pd['index'].isin(train_data_pd['index'].tolist())]
    test = response_data_pd.loc[response_data_pd['index'].isin(test_data_pd['index'].tolist())]
    # print(train[:2])
    # print(len(train_data_pd['index'].tolist()),len(test_data_pd['index'].tolist()),len(response_data_pd['index']))
    print(len(train),len(test))
    with open(f'{args.save_path}/responses/alpaca_7B_strqa_sampled_responses_train.json', 'w') as outfile:
        json.dump(train.to_dict(orient='list'), outfile)
    with open(f'{args.save_path}/responses/alpaca_7B_strqa_sampled_responses_test.json', 'w') as outfile:
        json.dump(test.to_dict(orient='list'), outfile)

    # with open(f'{args.save_path}/responses/gemma_2B_strqa_baseline_responses_train.json', 'r') as read_file:
    #     greedy_train_data = json.load(read_file)
    # with open(f'{args.save_path}/responses/gemma_2B_strqa_sampled_responses_train.json', 'r') as read_file:
    #     sampled_train_data = json.load(read_file)
    # # print(len(sampled_train_data['is_correct']))
    # # print(sampled_train_data.keys())
    # result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    # for i in range(len(sampled_train_data['is_correct'])):
    #     if len(sampled_train_data['is_correct'][i])>0:
    #         # print(sampled_train_data['is_correct'][i], [greedy_train_data['is_correct'][i]])
    #         result_dict['is_correct'].append(sampled_train_data['is_correct'][i] + [greedy_train_data['is_correct'][i]])
    #         result_dict['model_answer'].append(sampled_train_data['model_answer'][i] + [greedy_train_data['model_answer'][i]])
    #         result_dict['model_completion'].append(sampled_train_data['model_completion'][i] + [greedy_train_data['model_completion'][i]])
    #         result_dict['full_input_text'].append(sampled_train_data['full_input_text'][i] + [greedy_train_data['full_input_text'][i]])
    
    # print(len(result_dict['is_correct'])) # 1831
    # with open(f'{args.save_path}/responses/gemma_2B_strqa_sampledplus_responses_train.json', 'w') as f:
    #         json.dump(result_dict, f)
    ##

    # ##gsm8k
    # with open(f'{args.save_path}/responses/llama3.1_8B_Instruct_gsm8k_greedy_responses_train5000.json', 'r') as read_file:
    #     greedy_train_data = json.load(read_file)
    # with open(f'{args.save_path}/responses/llama3.1_8B_Instruct_gsm8k_sampled_responses_train5000.json', 'r') as read_file:
    #     sampled_train_data = json.load(read_file)
    # # print(len(sampled_train_data['is_correct']))
    # # print(sampled_train_data.keys())
    # result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    # for i in range(len(sampled_train_data['is_correct'])):
    #     if len(sampled_train_data['is_correct'][i])>0:
    #         # print(sampled_train_data['is_correct'][i], [greedy_train_data['is_correct'][i]])
    #         greedy_i = ''
    #         for j in range(len(greedy_train_data['full_input_text'])):
    #             if greedy_train_data['full_input_text'][j]==sampled_train_data['full_input_text'][i][0]: greedy_i=j # needed to match rows when generated using parallel gpus
    #         if greedy_i == '': print(i,sampled_train_data['full_input_text'][i])
    #         result_dict['is_correct'].append(sampled_train_data['is_correct'][i] + [greedy_train_data['is_correct'][greedy_i]])
    #         result_dict['model_answer'].append(sampled_train_data['model_answer'][i] + [greedy_train_data['model_answer'][greedy_i]])
    #         result_dict['model_completion'].append(sampled_train_data['model_completion'][i] + [greedy_train_data['model_completion'][greedy_i]])
    #         result_dict['full_input_text'].append(sampled_train_data['full_input_text'][i] + [greedy_train_data['full_input_text'][greedy_i]])
    
    # print(len(result_dict['is_correct'])) # 1999
    # with open(f'{args.save_path}/responses/llama3.1_8B_Instruct_gsm8k_sampledplus_responses_train5000.json', 'w') as f:
    #         json.dump(result_dict, f)

    ## this was not needed since start_at doesn't apply to gsm8k in get_prompt_responses_factual.py
    # result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    # for end in [2000,5000]:
    #     with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_greedy_responses_train{end}.json', 'r') as read_file:
    #         train_data = json.load(read_file)
    #         result_dict['is_correct'] += train_data['is_correct']
    #         result_dict['model_answer'] += train_data['model_answer']
    #         result_dict['model_completion'] += train_data['model_completion']
    #         result_dict['full_input_text'] += train_data['full_input_text']
    # print(len(result_dict['is_correct']))
    # with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_greedy_responses_train5000.json', 'w') as f:
    #         json.dump(result_dict, f)

    # with open(f'{args.save_path}/responses/hl_llama_7B_gsm8k_greedy_responses_train5000.json', 'r') as read_file:
    #     train_data = json.load(read_file)
    # print(len(train_data['is_correct']))
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