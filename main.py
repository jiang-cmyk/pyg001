import os.path as osp
import sys
import torch
import numpy as np


from torch_geometric.nn import Node2Vec

from sklearn.metrics import f1_score, roc_auc_score

from dataload import pred_triple

# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import Data
# from tqdm import tqdm
# import torch_geometric.transforms as T

def main():
    dataset = 'JF17k'#FB15k    FB15k-237    JF17k humans_wikidata
    print('model: Node2vec \t dataset: ', dataset)
    # osp.dirname(osp.realpath(__file__)), 
    path = osp.join('.', 'data', dataset)
    
    
    # test_data_r = FB15k_237(path, split='test')[0]
    # val_data_r = FB15k_237(path, split='val')[0]
    # train_data_r = FB15k_237(path, split'train')[0]
    #
    # edge_index = torch.cat((test_data_r.edge_index, val_data_r.edge_index, train_data_r.edge_index), dim=-1)
    # graph = Data(edge_index=edge_index, num_nodes=train_data_r.num_nodes)
    #
    # split = T.RandomLinkSplit(
    #     num_val=0.15,
    #     num_test=0.3,
    #     is_undirected=True,
    #     add_negative_train_samples=False,
    #     neg_sampling_ratio=1.0,
    #     )
    #
    # train_data, val_data, test_data = split(graph)
    # print(train_data)
    # print(val_data)
    # Data(edge_index=[2, 298446], num_nodes=14951, edge_label=[149223], edge_label_index=[2, 149223])
    # Data(edge_index=[2, 298446], num_nodes=14951, edge_label=[81392], edge_label_index=[2, 81392])
    
    
    train_data = pred_triple(path, split='train')[0]
    val_data = pred_triple(path, split='val')[0]
    test_data = pred_triple(path, split='test')[0]
    
    
    
    
    # print(train_data)
    # print(test_data)
    # print(val_data)
    max_index = torch.max(train_data['edge_label_index'][0].max(),train_data['edge_label_index'][1].max())
    mm = [False if val_data.edge_label_index[0][ind] > max_index  or val_data.edge_label_index[1][ind] > max_index else True for ind in range(len(val_data.edge_label_index[0]))]
    nn = [False if test_data.edge_label_index[0][ind] > max_index  or test_data.edge_label_index[1][ind] > max_index else True for ind in range(len(test_data.edge_label_index[0]))]
    val_data.edge_index = torch.cat((val_data.edge_label_index[0][mm], val_data.edge_label_index[1][mm])).reshape(2, -1)
    val_data.edge_label = val_data.edge_label[mm]
    test_data.edge_index = torch.cat((test_data.edge_label_index[0][nn], test_data.edge_label_index[1][nn])).reshape(2, -1)
    test_data.edge_label = test_data.edge_label[nn]
    
    # test_data.edge_index[0] = test_data.edge_label_index[0][nn]
    # val_data.edge_index[1] = val_data.edge_label_index[1][mm]
    # test_data.edge_index[1] = test_data.edge_label_index[1][nn]
    
    print(train_data)
    print(test_data)
    print(val_data)
    print(train_data.edge_type.max())
    # print(train_data['edge_label_index'][0].max(),train_data['edge_label_index'][1].max())
    # print(test_data['edge_index'][0].max(),test_data['edge_index'][1].max())
    # print(val_data['edge_index'][0].max(),val_data['edge_index'][1].max())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
                    train_data.edge_label_index,
                    # torch.cat((train_data.edge_label_index, test_data.edge_label_index), dim=1), 
                     embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    
    num_workers = 0 if sys.platform.startswith('win') else 4
    # print(num_workers)
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.008)
    
    
    def decode(z, edge_label_index):
        # print(edge_label_index.shape)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
 
    
    
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
        # mm = [True if data.edge_label_index[0][ind]< len(z)  and data.edge_label_index[1][ind] <len(z) else False for ind in range(len(data.edge_label_index[0]))]
        out = decode(z, data.edge_index).view(-1).sigmoid()
        y_pred = np.around(out.cpu().numpy(),0).astype(int) 
        # print()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()), f1_score(data.edge_label.cpu().numpy(), y_pred)
    
    
    for epoch in range(1, 100):
        loss = train()
        val_auc, val_f1 = link_pre(val_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
    
    test_auc, test_f1= link_pre(test_data)
    print(f'Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}')
    
    
if __name__ == "__main__":
    main()
    
    