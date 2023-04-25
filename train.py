from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import os.path as osp
import sys
import torch
from sklearn.manifold import TSNE

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from FB15k_237 import FB15k_237

import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec

def main():
    
    # dataset = 'FB15k-237'
    # # osp.dirname(osp.realpath(__file__)), 
    # path = osp.join('.', 'data', dataset)
    #
    # train_data = FB15k_237(path, split='train')[0]
    # val_data = FB15k_237(path, split='val')[0]
    # test_data = FB15k_237(path, split='test')[0]
    #
    #
    # print(train_data)
    # print(test_data)
    # print(val_data)
    
    dataset = 'FB15k_237'
    path = osp.join('.', 'data', dataset)
   
    test_data_r = FB15k_237(path, split='test')[0]
    val_data_r = FB15k_237(path, split='val')[0]
    train_data_r = FB15k_237(path, split='train')[0]
    
    edge_index = torch.cat((test_data_r.edge_index, val_data_r.edge_index, train_data_r.edge_index), dim=-1)
    graph = Data(edge_index=edge_index, num_nodes=train_data_r.num_nodes)

    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        )


    train_data, val_data, test_data = split(graph)
    print(test_data)
    print(val_data)
    print(train_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
                    train_data.edge_label_index,
                     # torch.cat((train_data.edge_label_index, test_data.edge_label_index), dim=1), 
                     embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    
    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)    
    
    @torch.no_grad()
    def link_pre(data):   
        model.eval()
        z = model()
 #       mm = [True if data.edge_label_index[0][ind]< len(z)  and data.edge_label_index[1][ind] <len(z) else False for ind in range(len(data.edge_label_index[0]))]
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    
    
    for epoch in range(1, 100):
        loss = train()
        val_auc = link_pre(val_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
    
    test_auc = link_pre(test_data)
    print(f'Test AUC: {test_auc:.4f}')
    
    
if __name__ == "__main__":
    main()
    
    
    