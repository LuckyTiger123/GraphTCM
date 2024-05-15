import torch
import argparse
from torch import Tensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch implementation of downstream adaptation.')

parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--layer_num', type=int, default=2, help='layer number')
parser.add_argument('--enhance', action='store_true', help='enhancing the embedding or not')
parser.add_argument('--device_num', type=int, default=0, help='device number')
parser.add_argument('--epoch_num', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.3, help='dropping rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='test dataset')
parser.add_argument('--train_round', type=int, default=5, help='training round number')
parser.add_argument('--path', type=str, help='embedding path')
args = parser.parse_args()

# Random seed setting.
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device setting.
device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

# Loading the dataset.
dataset = PygNodePropPredDataset(name=args.dataset, root='./data', transform=T.ToSparseTensor())
data = dataset[0]
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
evaluator = Evaluator(name='ogbn-arxiv')

# Loading the embeddings.
H = torch.load(args.path, map_location=torch.device('cpu'))
H = H.to(device=device)

# Enhancing the embeddings.
if args.enhance:
    H = torch.concat([H, data.x], dim=1)


# Model
class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, layer_num):
        super(MLP, self).__init__()
        self.layer = torch.nn.ModuleList()
        if layer_num == 1:
            self.layer.append(torch.nn.Linear(H.size(-1), dataset.num_classes))
        else:
            self.layer.append(torch.nn.Linear(H.size(-1), hidden_dim))
            self.layer.append(torch.nn.ReLU())
            for i in range(layer_num - 2):
                self.layer.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.layer.append(torch.nn.ReLU())
            self.layer.append(torch.nn.Linear(hidden_dim, dataset.num_classes))

    def forward(self, x: Tensor):
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        for layer in self.layer:
            x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layer:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()


model = MLP(args.hidden_dim, args.layer_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(H)
    loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    print('The train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(H)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


test_acc_list = []
for round in range(args.train_round):
    print('For the {} round'.format(round))
    best_val_acc = test_acc = 0
    model.reset_parameters()
    for epoch in range(args.epoch_num):
        print('---------------------------------------------------------------')
        print('For the {} epoch'.format(epoch))
        train()
        train_acc, val_acc, current_test_acc = test()
        print(
            'The train acc is {}, the val acc is {}, the test acc is {}.'.format(train_acc, val_acc, current_test_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = current_test_acc
    test_acc_list.append(test_acc)
acc_avg = float(np.average(test_acc_list))
acc_std = float(np.std(test_acc_list))
print('The avg acc is {}, and the std is {}.'.format(acc_avg, acc_std))
print('Mission completes.')
