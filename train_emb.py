import torch
import torch.nn as nn
import argparse

from module import Combiner, GraphTCM

parser = argparse.ArgumentParser(description='PyTorch implementation for training the representations.')

parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--device_num', type=int, default=0, help='device number')
parser.add_argument('--epoch_num', type=int, default=500, help='epoch number')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
parser.add_argument('--path', type=str, help='path for the trained GraphTCM model')
parser.add_argument('--target', type=str, default='zeros', help='training target (ones or zeros)')
parser.add_argument('--train_method', type=str, default='naive_agg', help='training method')
args = parser.parse_args()

# Loading embeddings from different methods.
h1 = torch.load('./emb/{}/GraphComp.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h2 = torch.load('./emb/{}/AttributeMask.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h3 = torch.load('./emb/{}/GAE.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h4 = torch.load('./emb/{}/EdgeMask.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h5 = torch.load('./emb/{}/NodeProp.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h6 = torch.load('./emb/{}/DisCluster.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h7 = torch.load('./emb/{}/DGI.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
h8 = torch.load('./emb/{}/SubgCon.pkl'.format(args.dataset), map_location=torch.device('cpu')).unsqueeze(0)
H = torch.cat((h1, h2, h3, h4, h5, h6, h7, h8), dim=0)

# Random seed setting.
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device setting.
device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

# Training target.
if args.target == 'ones':
    y_target = torch.ones(1, 8).to(device)
elif args.target == 'zeros':
    y_target = torch.zeros(1, 8).to(device)

# Loading GraphTCM.
cor_model = GraphTCM(input_size=H.size(-1), hidden_size=8, pooling='mean')
cor_model.load_state_dict(torch.load(args.path, map_location=torch.device('cpu')))
cor_model.to(device)
H = H.to(device)

model = Combiner(base_number=H.size(0), hidden_dim=H.size(-1), combine_style=args.train_method).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
# cal_loss = nn.L1Loss()
cal_loss = nn.MSELoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(H)
    cor = cor_model.predict(H, out.unsqueeze(0))
    loss = cal_loss(cor, y_target)
    loss.backward()
    optimizer.step()
    print('The train loss is {}'.format(loss.item()))
    return float(loss.item())


@torch.no_grad()
def valid():
    model.eval()
    out = model(H)
    cor = cor_model.predict(H, out.unsqueeze(0))
    print('The final correlation is {}'.format(cor))
    torch.save(out, './model/GraphTCM_enhanced_emb_{}.pkl'.format(args.dataset))
    return


min_loss = 100000
for epoch in range(args.epoch_num):
    print('---------------------------------------------')
    print('The {} epoch'.format(epoch))
    train_loss = train()
    if train_loss < min_loss:
        min_loss = train_loss
        torch.save(model.state_dict(), './model/GraphTCM_combiner_{}.pkl'.format(args.dataset))
        print('The model is saved successfully.')

model.load_state_dict(
    torch.load('./model/GraphTCM_combiner_{}.pkl'.format(args.dataset), map_location=torch.device('cpu')))
model.to(device)
valid()
print('Mission completed!')
