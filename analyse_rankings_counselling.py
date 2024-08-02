import os
import numpy as np
import json
import pandas as pd
import torch
from scipy.stats import rankdata
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


    llama_token_logprobs = np.load(f'{args.save_path}/features/counselling_wudata_llama_7B_token_logprobs.pkl',allow_pickle=True)
    gpt_token_logprobs_df = pd.read_excel(f'{args.save_path}/features/token_logprobs.xlsx')
    sampled_responses = pd.read_excel(file_path+'annotations_filtered.xlsx', index_col=0, sheet_name='Sheet2')

    groundtruth = []
    for context_id in sampled_responses.index:
        context_truth = []
        for resp_id in [1,2,3,4,5,6,7,8,9]:
            if sampled_responses['annotation'+str(resp_id)][context_id]=='Good':
                context_truth.append('Resp'+str(resp_id))
        groundtruth.append(context_truth)
    
    llama_rankings, gpt_rankings = [], []
    for i,row in gpt_token_logprobs_df.iterrows():
        llama_scores, gpt_scores = [], []
        llama_iterator = 0
        for resp_id in ['Resp1','Resp2','Resp3','Resp4','Resp5','Resp6','Resp7','Resp8','Resp9']:
            try:
                gpt_scores.append(np.mean(row[resp_id]))
                llama_scores.append(llama_token_logprobs[llama_iterator])
                llama_iterator += 1
            except TypeError: # Some prompts have fewer samples
                continue
        # rankdata(llama_scores)
        # rankdata(gpt_scores)
        # break
    
    assert len(llama_token_logprobs)==llama_iterator


if __name__ == '__main__':
    main()