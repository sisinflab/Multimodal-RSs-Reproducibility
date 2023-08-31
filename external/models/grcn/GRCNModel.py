from abc import ABC

from .GraphRefiningLayer import GraphRefiningLayer
from .GraphConvolutionalLayer import GraphConvolutionalLayer
import torch
import numpy as np
import random
from torch_geometric.utils import dropout_adj


class EGCN(torch.nn.Module):
    def __init__(self,
                 num_user,
                 num_item,
                 dim_E,
                 aggr_mode,
                 has_act,
                 has_norm,
                 num_layers):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.num_layers = num_layers
        for n in range(num_layers):
            self.__setattr__(f'conv_embed_{n + 1}', GraphConvolutionalLayer(has_act=has_act, aggr_mode=aggr_mode))

    def forward(self, adj, weight_vector):
        x = [torch.nn.functional.normalize(self.id_embedding) if self.has_norm else self.id_embedding]

        for layer in range(self.num_layers):
            x += self.__getattr__(f'conv_embed_{layer + 1}')(x[layer], adj, weight_vector)

        return torch.sum(x)


class CGCN(torch.nn.Module):
    def __init__(self,
                 features,
                 num_user,
                 num_item,
                 dim_C,
                 aggr_mode,
                 num_routing,
                 has_act,
                 has_norm,
                 rows,
                 ptr,
                 ptr_full,
                 device,
                 is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GraphRefiningLayer(rows, has_act, ptr, ptr_full)
        self.is_word = is_word
        self.device = device

        if is_word:
            self.word_tensor = torch.LongTensor(features).to(self.device)
            self.features = torch.nn.Embedding(torch.max(features[1]) + 1, dim_C)
            torch.nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = torch.nn.Linear(self.dim_feat, self.dim_C)

    def forward(self, adj, adj_user):

        if self.is_word:
            features = torch.tensor(scatter_('mean', self.features(self.word_tensor[1]), self.word_tensor[0])).to(self.device)
        else:
            features = torch.nn.functional.leaky_relu(self.MLP(self.features))

        if self.has_norm:
            preference = torch.nn.functional.normalize(self.preference)
            features = torch.nn.functional.normalize(features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            x_hat_1 = self.conv_embed_1(x, adj_user)
            preference = preference + x_hat_1[:self.num_user]

            if self.has_norm:
                preference = torch.nn.functional.normalize(preference)

        x = torch.cat((preference, features), dim=0)

        x_hat_1 = self.conv_embed_1(x, adj)

        if self.has_act:
            x_hat_1 = torch.nn.functional.leaky_relu_(x_hat_1)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 num_layers,
                 num_routings,
                 modalities,
                 aggregation,
                 weight_mode,
                 pruning,
                 has_act,
                 has_norm,
                 fusion_mode,
                 multimodal_features,
                 adj,
                 adj_user,
                 rows,
                 ptr,
                 ptr_full,
                 random_seed,
                 dropout,
                 name="GRCN",
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
        self.weight_mode = weight_mode
        self.fusion_mode = fusion_mode
        self.weight = torch.tensor([[1.0], [-1.0]]).cuda()
        self.l_w = l_w
        self.dropout = dropout
        self.modalities = modalities
        self.lr = learning_rate
        self.n_layers = num_layers

        self.adj = adj
        self.adj_user = adj_user
        self.id_gcn = EGCN(num_users, num_items, embed_k, aggregation, has_act, has_norm, num_layers)

        self.pruning = pruning

        # multimodal features
        for m_id, m in enumerate(modalities):
            self.__setattr__(f'{m}_feat', torch.from_numpy(multimodal_features[m_id]).type(torch.float32).to(self.device))
            self.__setattr__(f'{m}_gcn', CGCN(self.__getattribute__(f'{m}_feat'),
                                              num_users,
                                              num_items,
                                              embed_k_multimod,
                                              aggregation,
                                              num_routings,
                                              has_act,
                                              has_norm,
                                              rows=rows,
                                              ptr=ptr,
                                              ptr_full=ptr_full,
                                              device=self.device
                                              ))

        self.model_specific_conf = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.rand((num_users + num_items, len(modalities))))
        ).to(self.device)
        self.result = torch.nn.init.xavier_normal_(torch.rand((num_users + num_items, embed_k))).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': self.learning_rate}])

    def propagate_embeddings(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.adj, p=self.dropout)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep

        if self.a_feat is not None:
            num_modal += 1
            a_rep, weight_a = self.a_gcn(edge_index)
            if weight is None:
                weight = weight_a
                content_rep = a_rep
            else:
                content_rep = torch.cat((content_rep, a_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_a
                else:
                    weight = torch.cat((weight, weight_a), dim=1)

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t
                conetent_rep = t_rep
            else:
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)

        if self.weight_mode == 'mean':
            weight = weight / num_modal

        elif self.weight_mode == 'max':
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]),
                                   dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep + v_rep + a_rep + t_rep) / 4

        self.result = representation
        return representation

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.user_embeddings[start: stop].to(self.device),
                            torch.transpose(self.item_embeddings.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))

        loss = -torch.mean(torch.log(torch.sigmoid(xu_pos - xu_neg)))
        reg_content_loss = torch.sum(torch.stack(
            [self.Gum[m][np.concatenate([user[:, 0], user[:, 0]])].pow(2).mean() for m in self.modalities]))
        reg_loss = self.l_w * ((self.Gu.weight[np.concatenate([user[:, 0], user[:, 0]])].pow(2) +
                                self.Gi.weight[np.concatenate([pos[:, 0], neg[:, 0]])].pow(2)).mean())
        loss += (reg_loss + reg_content_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
