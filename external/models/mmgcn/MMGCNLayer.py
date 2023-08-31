import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
import random
import numpy as np
from torch_sparse import matmul


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', random_seed=42, **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, device, dim_latent=None, random_seed=42):
        super(GCN, self).__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = torch.nn.init.xavier_normal_(
                torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            self.MLP = torch.nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            torch.nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = torch.nn.Linear(self.dim_latent, self.dim_id)
            torch.nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = torch.nn.Linear(self.dim_latent + self.dim_id,
                                            self.dim_id) if self.concate else torch.nn.Linear(
                self.dim_latent, self.dim_id)
            torch.nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = torch.nn.init.xavier_normal_(
                torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            torch.nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = torch.nn.Linear(self.dim_feat, self.dim_id)
            torch.nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = torch.nn.Linear(self.dim_feat + self.dim_id,
                                            self.dim_id) if self.concate else torch.nn.Linear(
                self.dim_feat, self.dim_id)
            torch.nn.init.xavier_normal_(self.g_layer1.weight)

        for n in range(1, num_layer):
            self.__setattr__(f'conv_embed_{n + 1}', BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode))
            torch.nn.init.xavier_normal_(self.__getattr__(f'conv_embed_{n + 1}').weight)
            self.__setattr__(f'linear_layer{n + 1}', torch.nn.Linear(self.dim_id, self.dim_id))
            torch.nn.init.xavier_normal_(self.__getattr__(f'linear_layer{n + 1}').weight)
            self.__setattr__(f'g_layer{n + 1}', torch.nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else torch.nn.Linear(
                self.dim_id,
                self.dim_id))

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = torch.nn.functional.normalize(x).to(self.device)

        for n in range(self.num_layer):
            h = torch.nn.functional.leaky_relu(self.__getattr__(f'conv_embed_{n + 1}')(x, self.edge_index))  # equation 1
            x_hat = torch.nn.functional.leaky_relu(self.__getattr__(f'linear_layer{n + 1}')(x)) + id_embedding if self.has_id else torch.nn.functional.leaky_relu(
                self.__getattr__(f'linear_layer{n + 1}')(x))  # equation 5
            x = torch.nn.functional.leaky_relu(self.__getattr__(f'g_layer{n + 1}')(torch.cat((h, x_hat), dim=1))) if self.concate else torch.nn.functional.leaky_relu(
                self.__getattr__(f'g_layer{n + 1}')(h) + x_hat)

        return x
