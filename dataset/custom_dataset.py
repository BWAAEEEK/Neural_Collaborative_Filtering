from torch.utils.data import Dataset
import torch
import pickle
import random
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, interaction, user_list, movie_list):
        super().__init__()

        self.interaction = interaction
        self.user_list = user_list
        self.movie_list = movie_list

        # construct dataset
        self.data = []
        for user, items in tqdm(self.interaction.items()):
            for item in items:
                prob = random.random()

                if prob > 0.8:
                    self.data.append(self.negative_sampling(user))
                else:
                    self.data.append(self.sampling(user))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = self.data[idx]

        return {key: torch.tensor(value) for key, value in tmp.items()}

    def sampling(self, user):
        item = random.choice(self.interaction[user])

        return {"user": user, "item": item, "label": [0]}

    def negative_sampling(self, user):
        item = 0
        while True:
            item = random.choice(self.movie_list)

            if item not in self.interaction[user]:
                break

        return {"user": user, "item": item, "label": [0]}
