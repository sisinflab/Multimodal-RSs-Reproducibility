import random
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index, seed):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_item))
        random.seed(seed)

    def __len__(self):
        return self.edge_index.shape[1]

    def __getitem__(self, index):
        user, pos_item = self.edge_index[:, index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item])
