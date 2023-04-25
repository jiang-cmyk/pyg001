from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import os.path as osp
import sys
import torch
from sklearn.manifold import TSNE

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from tqdm import tqdm
from dataload import FB15k_237

from torch_geometric.data import Data

import torch_geometric.transforms as T


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def encode(self, x, edge_index):
        # print(x.shape, edge_index.shape)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
 
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
 
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
     
 
def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs=100
 ):
 
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
 
        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
 
        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
 
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
 
        val_auc = eval_link_predictor(model, val_data)
 
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
 
    return model
 
 
@torch.no_grad()
def eval_link_predictor(model, data):
 
    model.eval()
    z = model.encode(data.x, data.edge_index)
    print(z.shape)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
     
     
 
split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
 )


dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
graph = dataset[0]
train_data, val_data, test_data = split(graph)


print(train_data)
print(val_data)
print(test_data)
# print(dataset)
# print(dataset.num_features)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = Net(dataset.num_features, 128, 64)#.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")