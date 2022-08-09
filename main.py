import argparse
import os
import time
import pickle
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from model import NCF
from trainer import Trainer
from dataset import CustomDataset

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--interaction_path", type=str, default="./interaction.pkl")
    args.add_argument("--user_path", type=str, default="./user_list.pkl")
    args.add_argument("--movie_path", type=str, default="./movie_list.pkl")

    args.add_argument("--user_len", type=int, default=6040)
    args.add_argument("--item_len", type=int, default=3952)

    args.add_argument("--embed_size", type=int, default=100)
    args.add_argument("--hidden_size_1", type=int, default=1024)
    args.add_argument("--hidden_size_2", type=int, default=128)
    args.add_argument("--hidden_size_3", type=int, default=64)


    args.add_argument("--num_workers", type=int, default=5)
    args.add_argument("--batch_size", type=int, default=1000)
    args.add_argument("--epochs", type=int, default=10)

    args.add_argument("--learning_rate", type=float, default=0.001)


    args.add_argument("--seed", type=int, default=42)

    args = args.parse_args()

    fix_seed(args.seed)

    print("\n Loading Data ...")
    with open(args.interaction_path, "rb") as f:
        interaction = pickle.load(f)

    with open(args.user_path, "rb") as f:
        user_list = pickle.load(f)

    with open(args.movie_path, "rb") as f:
        item_list = pickle.load(f)

    print()
    print("size of user :", args.user_len, "size of item:", args.item_len)

    print("\n Loading Custom Dataset")
    dataset = CustomDataset(interaction, user_list, item_list)

    print("\n Creating Data Loader")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    print("\n Building NCF Model")
    model = NCF(args)

    print("\n Building Model Trainer")
    trainer = Trainer(model, dataloader)

    time.sleep(0.5)
    print("\n--- Training Start ---")
    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        trainer.train(epoch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        trainer.save(args.epochs, args.output_path)
