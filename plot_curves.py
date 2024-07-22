import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import json
import argparse
from matplotlib import pyplot as plt
import wandb

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='')
    parser.add_argument('dataset_name', type=str, default='')
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--probes_file_name',type=str, default='')
    parser.add_argument('--using_act',type=str, default='')
    parser.add_argument('--token',type=str, default='')
    parser.add_argument('--method',type=str, default='')
    parser.add_argument('--bs',type=int, default=None)
    parser.add_argument('--lr',type=float, default=None)
    args = parser.parse_args()

    device = 0

    val_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_loss.npy', allow_pickle=True).item()[0]
    train_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_train_loss.npy', allow_pickle=True).item()[0]
    try:
        supcon_train_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()[0]
    except FileNotFoundError:
        supcon_train_loss = []

    val_loss = val_loss[-1] # Last layer only
    train_loss = train_loss[-1] # Last layer only
    if len(supcon_train_loss)>0: supcon_train_loss = supcon_train_loss[-1] # Last layer only

    if len(val_loss)!=len(train_loss):
        train_loss_by_epoch = []
        batches = int(len(train_loss)/len(val_loss))
        start_at = 0
        for epoch in range(len(val_loss)):
            train_loss_by_epoch.append(sum(train_loss[start_at:(start_at+batches)]))
            start_at += batches
        train_loss = train_loss_by_epoch

    print(len(val_loss))
    print(len(train_loss))
    if len(supcon_train_loss)>0: print(supcon_train_loss.shape)
    
    plt.subplot(1, 3, 1)
    plt.plot(val_loss)
    plt.xlabel("epoch")
    plt.ylabel( "loss")
    plt.title('val loss')
    plt.subplot(1, 3, 2)
    plt.plot(train_loss)
    plt.xlabel("epoch")
    plt.title('train ce loss')
    plt.subplot(1, 3, 3)
    plt.plot(supcon_train_loss)
    plt.xlabel("epoch")
    plt.title('train supcon loss')
    plt.savefig(f'{args.save_path}/testfig2.png')

    # wandb.init(
    # project="LLM-Hallu-Detection",
    # config={
    # "run_name": args.probes_file_name,
    # "model": args.model_name,
    # "dataset": args.dataset_name,
    # "act_type": args.using_act,
    # "token": args.token,
    # "method": args.method,
    # "bs": args.bs,
    # "lr": args.lr,
    # }
    # )
    # wandb.log({'chart': plt})


    
if __name__ == '__main__':
    main()