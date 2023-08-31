from abc import ABC

from .MMGCNLayer import GCN
import torch
import numpy as np
import random


class MMGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 batch_size,
                 l_w,
                 num_layers,
                 modalities,
                 aggregation,
                 concatenation,
                 has_id,
                 multimodal_features,
                 adj,
                 random_seed,
                 name="MMGCN",
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
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.modalities = modalities
        self.aggr_mode = aggregation
        self.concate = concatenation
        self.n_layers = num_layers
        self.has_id = has_id
        self.batch_size = batch_size

        self.edge_index = adj.to(self.device)

        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)

        for m_id, m in enumerate(self.modalities):
            self.__setattr__(f'{m}_feat', torch.tensor(multimodal_features[m_id], dtype=torch.float).to(self.device))
            self.__setattr__(f'{m}_gcn', GCN(
                self.edge_index,
                self.batch_size,
                self.num_users,
                self.num_items,
                self.__getattribute__(f'{m}_feat').size(1),
                self.embed_k,
                self.aggr_mode,
                self.concate,
                num_layer=self.n_layers,
                has_id=self.has_id,
                device=self.device,
                dim_latent=self.embed_k_multimod[m_id])
            )

        self.id_embedding = torch.nn.init.xavier_normal_(torch.rand((self.num_users + self.num_items, self.embed_k), requires_grad=True)).to(self.device)
        self.result = torch.nn.init.xavier_normal_(torch.rand((self.num_users + self.num_items, self.embed_k))).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': self.learning_rate}])

    def forward(self):
        representation = 0
        for m_id, m in enumerate(self.modalities):
            representation += self.__getattr__(f'{m}_gcn')(self.__getattribute__(f'{m}_feat'), self.id_embedding)
        representation /= len(self.modalities)

        self.result = representation
        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1) + self.num_users
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor] ** 2 + self.id_embedding[item_tensor] ** 2).mean() + (
                    self.__getattr__(f'{self.modalities[0]}_gcn').preference ** 2).mean()
        reg_loss = self.l_w * reg_embedding_loss
        return loss + reg_loss, reg_loss, loss, reg_embedding_loss, reg_embedding_loss

    @staticmethod
    def predict(gu, gi, **kwargs):
        return torch.matmul(gu, gi.t())

    def train_step(self, user_tensor, item_tensor):
        self.optimizer.zero_grad()
        loss, model_loss, reg_loss, weight_loss, entropy_loss = self.loss(user_tensor, item_tensor)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
