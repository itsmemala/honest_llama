import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import json
from matplotlib import pyplot as plt
import wandb

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--probes_file_name',type=str, default='')
    args = parser.parse_args()

    device = 0

    wandb.init(
    project="LLM-Hallu-Detection",
    config={
    "run_name": args.probes_file_name
    }
    )

    val_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_val_loss.npy', allow_pickle=True).item()
    train_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_train_loss.npy', allow_pickle=True).item()
    try:
        supcon_train_loss = np.load(f'{args.save_path}/probes/{args.probes_file_name}_supcon_train_loss.npy', allow_pickle=True).item()
    except FileNotFoundError:
        supcon_train_loss = []
    
    plt.subplot(1, 3, 1)
    plt.plot(val_loss)
    plt.xlabel("epoch")
    plt.ylabel( "loss")
    plt.subplot(1, 3, 2)
    plt.plot(train_loss)
    plt.xlabel("epoch")
    plt.ylabel( "loss")
    plt.subplot(1, 3, 3)
    plt.plot(supcon_train_loss)
    plt.xlabel("epoch")
    plt.ylabel( "loss")
    wandb.log({'chart': plt})


    
if __name__ == '__main__':
    main()