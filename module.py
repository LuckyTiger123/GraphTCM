from torch_geometric.nn.dense.linear import Linear
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F


# The implementation of GraphTCM.
class GraphTCM(nn.Module):
    def __init__(self, input_size, hidden_size, pooling='mean'):
        super(GraphTCM, self).__init__()

        # Linear transformations.
        self.w1 = Linear(input_size, hidden_size, weight_initializer='glorot')
        self.w2 = Linear(input_size, hidden_size, weight_initializer='glorot')

        # Pooling functions.
        if pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'sum':
            self.pooling = global_add_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        elif pooling == 'attention':
            self.pooling = AttentionalAggregation(Linear(hidden_size, 1))
        else:
            raise NotImplementedError('Pooling type {} is not implemented'.format(pooling))

        self.pooling_type = pooling

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()

    def forward(self, H):
        H0 = H
        embed1 = self.w1(H0)
        embed2 = self.w2(H0)

        if self.pooling_type == 'attention':
            h1 = None
            h2 = None
            for i in range(embed1.size(0)):
                item1 = self.pooling(embed1[i])
                item2 = self.pooling(embed2[i])
                if i == 0:
                    h1 = item1
                    h2 = item2
                else:
                    h1 = torch.cat((h1, item1), dim=0)
                    h2 = torch.cat((h2, item2), dim=0)
        else:
            h1 = self.pooling(embed1, batch=None)
            h2 = self.pooling(embed2, batch=None)

        # Compute the correlation score.
        score = torch.mm(h1, h2.t())
        score = score - torch.diag(score)
        cor = torch.exp(score)

        return cor

    def predict(self, H, Emb):
        embed1 = self.w1(Emb)
        embed2 = self.w2(H)
        embed3 = self.w1(H)

        if self.pooling_type == 'attention':
            h1 = None
            h2 = None
            h3 = None
            for i in range(embed1.size(0)):
                item1 = self.pooling(embed1[i])
                if i == 0:
                    h1 = item1
                else:
                    h1 = torch.cat((h1, item1), dim=0)

            for i in range(embed2.size(0)):
                item2 = self.pooling(embed2[i])
                if i == 0:
                    h2 = item2
                else:
                    h2 = torch.cat((h2, item2), dim=0)

            for i in range(embed3.size(0)):
                item3 = self.pooling(embed3[i])
                if i == 0:
                    h3 = item3
                else:
                    h3 = torch.cat((h3, item3), dim=0)

        else:
            h1 = self.pooling(embed1, batch=None)
            h2 = self.pooling(embed2, batch=None)
            h3 = self.pooling(embed3, batch=None)

        score = torch.mm(h1, h2.t()) - torch.diag(torch.mm(h3, h2.t()))
        cor = torch.exp(score)

        return cor


# The combiner to learn more effective representations.
class Combiner(nn.Module):
    def __init__(self, base_number, hidden_dim, combine_style='naive_agg'):
        super(Combiner, self).__init__()
        self.combine_style = combine_style
        if combine_style == 'naive_agg':
            self.p = nn.Parameter(torch.randn(base_number, 1, 1))
        elif combine_style == 'weighted_agg':
            self.p = nn.Parameter(torch.randn(base_number, 1, hidden_dim))
        else:
            raise NotImplementedError('Combine style {} is not implemented'.format(combine_style))

    def forward(self, H):
        alpha = F.softmax(self.p, dim=0)
        H = H * alpha
        result = torch.sum(H, dim=0)
        return result

