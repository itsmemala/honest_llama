import os
import numpy as np
import json
import pandas as pd
import torch
from ast import literal_eval
from scipy.stats import rankdata, kendalltau, spearmanr
import argparse

def main(): 
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name',type=str, default='')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()


    llama_token_logprobs = np.load(f'{args.save_path}/features/counselling_wudata_{args.model_name}_token_logprobs.pkl',allow_pickle=True)
    gpt_token_logprobs_df = pd.read_excel(f'{args.save_path}/features/token_logprobs.xlsx')
    tokens_df = pd.read_excel(f'{args.save_path}/features/tokens.xlsx')
    sampled_responses = pd.read_excel(f'{args.save_path}/responses/annotations_filtered.xlsx', index_col=0, sheet_name='Sheet2')

    def my_isnan(val):
        try:
            return isnan(float(val))
        except:
            return False

    def f(x):
        try:
            return literal_eval(str(x))
        except Exception as e:
            return []

    def get_token_logprobs(i,j,resp_logprobs,tokens):
        context_id = sampled_responses.index[j]
        resp = sampled_responses[sampled_responses.index==context_id]['response'+str(i)][context_id]
        if my_isnan(resp)==False:
            resp = ' '+str(resp)
            k = len(resp_logprobs)
            for k in range(1,len(resp_logprobs)+1):
                if ''.join(tokens[j][i-1][:k])==resp:
                    # print(j,i,k)
                    break
            return resp_logprobs[:k]
        else:
            return resp_logprobs
    
    tokens = [[] for j in range(len(tokens_df['Resp1']))]
    for i in [1,2,3,4,5,6,7,8,9]:
        tokens_df['Resp'+str(i)] = tokens_df['Resp'+str(i)].apply(lambda x: f(x))
        for j,t in enumerate(tokens_df['Resp'+str(i)].tolist()):
            tokens[j].append(t)

    for i in [1,2,3,4,5,6,7,8,9]:
        gpt_token_logprobs_df['Resp'+str(i)] = gpt_token_logprobs_df['Resp'+str(i)].apply(lambda x: f(x))
        gpt_token_logprobs_df['Resp'+str(i)] = [get_token_logprobs(i,j,resp_logprobs,tokens) for j,resp_logprobs in enumerate(gpt_token_logprobs_df['Resp'+str(i)].tolist())] # Since sometimes GPT continues the generation beyond the original response that we want probs for

    groundtruth = []
    for context_id in sampled_responses.index:
        context_truth = []
        for resp_id in [1,2,3,4,5,6,7,8,9]:
            if sampled_responses['annotation'+str(resp_id)][context_id]=='Good':
                context_truth.append('Resp'+str(resp_id))
        groundtruth.append(context_truth)

    # print(len(llama_token_logprobs)) # 78

    llama_rankings, gpt_rankings, corr1, corr2 = [], [], [], []
    llama_iterator = 0
    for i,row in gpt_token_logprobs_df.iterrows():
        llama_scores, gpt_scores = [], []
        for resp_id in ['Resp1','Resp2','Resp3','Resp4','Resp5','Resp6','Resp7','Resp8','Resp9']:
            if len(row[resp_id])>0:
                gpt_scores.append(np.mean(row[resp_id]))
                llama_scores.append(np.mean(llama_token_logprobs[llama_iterator]))
                llama_iterator += 1
            else: # Some prompts have fewer samples
                continue
        llama_rankings.append(rankdata(llama_scores))
        gpt_rankings.append(rankdata(gpt_scores))
        corr1.append(kendalltau(rankdata(llama_scores), rankdata(gpt_scores)))
        corr2.append(spearmanr(rankdata(llama_scores), rankdata(gpt_scores)))
    
    print('Avg rank correlation:',np.mean(corr1), np.mean(corr2))
    


if __name__ == '__main__':
    main()