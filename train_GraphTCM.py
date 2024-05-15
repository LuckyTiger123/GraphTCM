import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from module import GraphTCM

parser = argparse.ArgumentParser(description='PyTorch implementation for building the correlation.')

parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dimension')
parser.add_argument('--pooling', type=str, default='mean', help='pooling type')
parser.add_argument('--device_num', type=int, default=0, help='device number')
parser.add_argument('--epoch_num', type=int, default=2000, help='epoch number')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--valid_rate', type=float, default=0.1, help='validation rate')
parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
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

# Pre-calculated correlation values for different datasets.
if args.dataset == 'Cora':
    y = torch.tensor([[1, 0.996306525, 1.605174424, 4.027285278, 3.181937772, 1.661625053, 7.975674549, 3.006051552],
                      [1.031106625, 1, 1.606166444, 4.26620875, 3.266461734, 1.927162989, 8.039007448, 3.018195602],
                      [1.045533498, 0.972069987, 1, 3.205840362, 3.15831691, 1.469154306, 8.023598538, 1.313930466],
                      [0.997270112, 0.751533387, 1.489581955, 1, 2.827174901, 1.470506762, 7.969720071, 2.890676961],
                      [1.018686244, 0.82794508, 1.575802144, 3.468804891, 1, 1.696328541, 8.005601185, 2.874860589],
                      [1.014361075, 0.98504649, 1.601873651, 4.049493043, 3.174094795, 1, 7.986145857, 3.014038985],
                      [0.933354998, 0.95824256, 1.545733145, 1.596365708, 3.134710965, 1.204679288, 1, 2.98943336],
                      [0.93372477, 0.679169105, 0.963227901, 2.440406342, 2.94783907, 0.97863032, 5.798309471, 1]])
elif args.dataset == 'CiteSeer':
    y = torch.tensor([[1, 0.996789, 1.706823, 5.132208, 2.934456, 2.154893, 7.434356, 4.495874],
                      [1.026115, 1, 1.707687, 5.754158, 2.994363, 2.519208, 7.452168, 4.53976],
                      [1.04322, 0.97483, 1, 3.393287, 2.839168, 1.827404, 7.429283, 1.650947],
                      [1.000287, 0.886627, 1.641847, 1, 2.636388, 1.998794, 7.429923, 4.358282],
                      [1.018431, 0.888094, 1.657101, 4.689336, 1, 2.011317, 7.438052, 4.208889],
                      [1.009889, 0.992822, 1.704207, 5.462861, 2.83891, 1, 7.421023, 4.532055],
                      [0.92039, 0.958346, 1.641221, 1.742211, 2.836609, 1.437741, 1, 4.477043],
                      [0.927804, 0.600141, 0.870486, 2.352276, 2.473089, 1.162364, 3.846964, 1]])
elif args.dataset == 'PubMed':
    y = torch.tensor([[1, 0.999425428, 1.938253585, 2.351274946, 1.687327813, 1.15730592, 2.164432873, 1.082583963],
                      [1.026091793, 1, 1.938463498, 2.417573153, 1.702027141, 1.173942217, 2.183564563, 1.082594288],
                      [0.993789346, 0.819607124, 1, 1.783357256, 1.631350789, 0.745068931, 1.866658325, 0.645656026],
                      [1.041699094, 0.894928894, 1.739051179, 1, 1.227248317, 0.922912952, 2.156107085, 1.050105804],
                      [1.035627736, 0.964745371, 1.869833309, 1.485442942, 1, 1.069534238, 2.182300998, 1.041164262],
                      [1.019006126, 0.989471596, 1.926540349, 2.243960076, 1.663642779, 1, 2.181998387, 1.08157404],
                      [0.900571617, 0.948035411, 1.828845385, 1.129046921, 1.310836531, 0.833584151, 1, 1.076035302],
                      [0.908883188, 0.900056034, 1.702064565, 2.016352178, 1.512810004, 0.844788333, 2.171826475, 1]])
elif args.dataset == "Arxiv":
    y = torch.tensor([[1, 1.042930178, 0.998105547, 1.310744533, 1.33322452, 1.832097274, 1.016693045, 1.34532368],
                      [0.947719718, 1, 0.961138147, 1.204750369, 1.2792013, 1.319562039, 0.997917389, 1.26064251],
                      [2.470983557, 1.022931069, 1, 1.151451225, 1.318546509, 2.068981023, 1.009779177, 1.013206134],
                      [1.046243292, 0.961937118, 1.112686626, 1, 1.122882712, 1.085023352, 1.002787149, 0.959838119],
                      [1.028361505, 1.005794128, 0.981191022, 1.077747898, 1, 1.234425079, 0.979110311, 0.93174249],
                      [0.969603151, 1.012790355, 0.968869356, 1.212147563, 1.213704376, 1, 0.967057487, 1.319488297],
                      [1.650853899, 1.031368464, 1.221945072, 1.163487016, 1.319974269, 2.195486516, 1, 1.271383471],
                      [0.98292633, 0.714734167, 0.927006386, 1.047050419, 1.149160368, 1.273961301, 1.006379883, 1]])
elif args.dataset == "Computers":
    y = torch.tensor([[1, 1.148841549, 0.999999746, 1.120493549, 1.761971948, 1.15332591, 3.569510433, 4.151653082],
                      [0.998471917, 1, 1.222708269, 1.030214789, 1.686619281, 1.102593057, 3.550543478, 3.820532017],
                      [1, 1.148841549, 1, 1.120493549, 1.761971948, 1.15332591, 3.569308523, 4.151653082],
                      [1.336801542, 0.982683359, 4.72878314, 1, 1.304333249, 0.749756215, 3.221502134, 3.691634215],
                      [2.10395432, 1.047647277, 4.84443834, 0.876748412, 1, 0.801122457, 3.489446739, 3.701517432],
                      [0.995856339, 1.13953469, 1.695548733, 1.066403145, 1.684791017, 1, 3.549341596, 3.971238276],
                      [0.949632947, 1.132202805, 0.993382139, 0.799028057, 1.637694162, 0.783561892, 1, 4.148919415],
                      [0.952468368, 0.943864356, 0.69416979, 0.770239078, 1.006680151, 0.491123253, 2.390441943, 1]])
elif args.dataset == "Photo":
    y = torch.tensor([[1, 1.138829184, 1.177702519, 1.339981024, 2.440887082, 1.31310189, 4.722465681, 4.097485956],
                      [0.998398215, 1, 2.792797317, 1.288791441, 2.301210024, 1.116074864, 4.515586379, 3.600003297],
                      [1.000876851, 1.127031735, 1, 1.33105416, 2.101202706, 1.051594759, 3.951282041, 4.06309531],
                      [1.095647457, 0.960092093, 4.03112392, 1, 1.820207081, 0.933931143, 4.517723271, 3.894630652],
                      [2.107245929, 0.9996591, 4.74320399, 1.069044601, 1, 1.048411229, 4.621416201, 3.735351698],
                      [0.99295263, 1.089142595, 1.399030055, 1.302999462, 2.353445423, 1, 4.41429231, 3.760811614],
                      [0.947771108, 1.117049023, 1.168035958, 0.94710454, 2.299204058, 0.790368628, 1, 4.09395489],
                      [0.94276986, 0.872687134, 0.776850303, 0.844949185, 1.355453482, 0.47604273, 2.828205138, 1]])
else:
    raise ValueError("Invalid dataset name.")

# Random seed setting.
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device setting.
device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

# Splitting train set and valid set.
total_num = H.size(0) ** 2
perm = torch.randperm(total_num)
train_mask = perm[:int(total_num * (1 - args.valid_rate))]
valid_mask = perm[int(total_num * (1 - args.valid_rate)):]

H = H.to(device)
y = y.to(device)

model = GraphTCM(H.size(-1), args.hidden_dim, args.pooling).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
# loss_cal = nn.L1Loss()
loss_cal = nn.MSELoss()


# Drawing pic.
def heatmap(y, title):
    df = pd.DataFrame(y.numpy())
    df.columns = ['GraphComp', 'AttributeMask', 'GAE', 'EdgeMask', 'NodeProp', 'DisCluster', 'DGI', 'SubgCon']
    df.index = ['GraphComp', 'AttributeMask', 'GAE', 'EdgeMask', 'NodeProp', 'DisCluster', 'DGI', 'SubgCon']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.xlabel('Method')
    plt.ylabel('Method')
    plt.xticks(range(8), df.index, rotation=45)
    plt.yticks(range(8), df.columns)
    plt.imshow(df, cmap='Blues', vmin=0, vmax=4)
    plt.colorbar()
    plt.show()


def train():
    model.train()
    optimizer.zero_grad()
    cor = model(H)
    cor_train = cor.view(-1, 1)[train_mask]
    y_train = y.view(-1, 1)[train_mask]
    loss = loss_cal(cor_train, y_train)
    loss.backward()
    optimizer.step()
    print('The train loss is {}'.format(loss.item()))
    return float(loss.item())


@torch.no_grad()
def valid():
    model.eval()
    cor = model(H)
    cor_valid = cor.view(-1, 1)[valid_mask]
    y_valid = y.view(-1, 1)[valid_mask]
    eval_loss = loss_cal(cor_valid / y_valid, torch.ones(cor_valid.size(0), 1).to(device))
    print('The eval loss is {}'.format(eval_loss.item()))
    return float(eval_loss.item())


def test():
    model.load_state_dict(
        torch.load('./model/GraphTCM_{}.pkl'.format(args.dataset), map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    cor = model(H)
    heatmap(cor.cpu().detach(), '{}-Test Correlation'.format(args.dataset))


min_loss = 100000
for epoch in range(args.epoch_num):
    print('---------------------------------------------')
    print('The {} epoch'.format(epoch))
    train_loss = train()
    valid_loss = valid()
    if valid_loss < min_loss:
        min_loss = valid_loss
        torch.save(model.state_dict(), './model/GraphTCM_{}.pkl'.format(args.dataset))

print('The min valid MSE is {}'.format(min_loss))
test()
print('Mission Complete!')
