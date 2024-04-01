import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
# from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument("--activations_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise'}

    # Load activations
    layer=31
    activatiions = []
    for file_end in [(a*100,(a*100)+100) for a in range(int(args.len_dataset/100))]:    
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.activations_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
        act = torch.from_numpy(np.load(file_path,allow_pickle=True)[:,layer,:]).to(device)
        activations.append(act)
    activations = torch.stack(activations,axis=0) if args.token in ['answer_last','prompt_last','maxpool_all'] else torch.cat(activations,dim=0)

    # Elbow plot for optimum number of clusters
    Sum_of_squared_distances = []
    for num_clusters in [5,10,25,50,100,250,500,1000,2500,5000]:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(activations)
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K,Sum_of_squared_distances,'bx-')
    plt.xlabel('Number of clusters') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow curve - Cluster analysis')
    plt.savefig(f'{args.save_path}/figures/{args.model_name}_{args.activations_file_name}_{args.using_act}_{args.token}_kmeans_elbow.png')

if __name__ == '__main__':
    main()