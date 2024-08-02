import os
import numpy as np
import json
import pandas as pd
import torch
from scipy.stats import rankdata
import argparse

def f(x):
    try:
        return literal_eval(str(x))
    except Exception as e:
        return []

def get_token_logprobs(i,j,resp_logprobs):
    context_id = sampled_responses.index[j]
    resp = sampled_responses[sampled_responses.index==context_id]['response'+str(i)][context_id]
    if my_isnan(resp)==False:
        resp = ' '+resp
        for k in range(1,len(resp_logprobs)+1):
            if ''.join(tokens[j][i-1][:k])==resp:
                print(j,i,k)
                break
        return resp_logprobs[:k]
    else:
        return resp_logprobs

def main(): 
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()


    llama_token_logprobs = np.load(f'{args.save_path}/features/counselling_wudata_llama_7B_token_logprobs.pkl',allow_pickle=True)
    gpt_token_logprobs_df = pd.read_excel(f'{args.save_path}/features/token_logprobs.xlsx')
    sampled_responses = pd.read_excel(f'{args.save_path}/responses/annotations_filtered.xlsx', index_col=0, sheet_name='Sheet2')

    for i in [1,2,3,4,5,6,7,8,9]:
        df['Resp'+str(i)] = df['Resp'+str(i)].apply(lambda x: f(x))
        df['Resp'+str(i)] = [get_token_logprobs(i,j,resp_logprobs) for j,resp_logprobs in enumerate(df['Resp'+str(i)].tolist())]

    groundtruth = []
    for context_id in sampled_responses.index:
        context_truth = []
        for resp_id in [1,2,3,4,5,6,7,8,9]:
            if sampled_responses['annotation'+str(resp_id)][context_id]=='Good':
                context_truth.append('Resp'+str(resp_id))
        groundtruth.append(context_truth)

    # print(len(llama_token_logprobs)) # 78

    llama_rankings, gpt_rankings = [], []
    llama_iterator = 0
    for i,row in gpt_token_logprobs_df.iterrows():
        llama_scores, gpt_scores = [], []
        for resp_id in ['Resp1','Resp2','Resp3','Resp4','Resp5','Resp6','Resp7','Resp8','Resp9']:
            # try:
            gpt_scores.append(np.mean(row[resp_id]))
            llama_scores.append(np.mean(llama_token_logprobs[llama_iterator]))
            llama_iterator += 1
            # except TypeError: # Some prompts have fewer samples
            #     continue
        print(llama_scores, rankdata(llama_scores))
        print(gpt_scores, rankdata(gpt_scores))
        break
    


if __name__ == '__main__':
    main()