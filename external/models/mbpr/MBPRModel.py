from abc import ABC

import torch
import numpy as np
import random

from torch.nn.init import xavier_normal_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, torch.nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, torch.nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class MBPRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 multimodal_features,
                 modalities,
                 lr_sched,
                 random_seed,
                 name="MBPR",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.num_users, self.embed_k * len(modalities)))
        ).to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.num_items, self.embed_k))
        )

        # multimodal
        self.F = torch.from_numpy(multimodal_features).type(torch.float32).to(self.device)
        self.feature_shape = multimodal_features.shape[1]
        self.proj = torch.nn.Linear(in_features=self.feature_shape, out_features=self.embed_k)
        self.proj.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler = lr_sched
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.apply(xavier_normal_initialization)

    def propagate_embeddings(self, dropout=0.0):
        item_embeddings = self.proj(self.F).to(self.device)
        item_embeddings = torch.cat((self.Gi, item_embeddings), -1)

        user_e = torch.nn.functional.dropout(self.Gu, dropout)
        item_e = torch.nn.functional.dropout(item_embeddings, dropout)
        return user_e, item_e

    @staticmethod
    def predict(gu, gi, **kwargs):
        return torch.matmul(gu, gi.transpose(0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        user_e = gu[user, :]
        pos_e = gi[pos, :]
        neg_e = gi[neg, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = - torch.log(1e-10 + torch.sigmoid(pos_item_score - neg_item_score)).mean()
        reg_loss = self.l_w * (torch.norm(user_e, 2) +
                               torch.norm(pos_e, 2) +
                               torch.norm(neg_e, 2)) / np.array(user).shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
