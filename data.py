import os.path as osp


from torch_geometric.datasets import Planetoid
from FB15k_237 import FB15k_237
import torch
from torch_geometric.data import Data
from dataload import pred_triple
import torch_geometric.transforms as T
# dataset1 = 'Cora'
# path1 = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset1)
# dataset1 = Planetoid(path1, dataset1)
# data1 = dataset1[0]
# print(data1)
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

#
# dataset = 'FB15k_237'
# path = osp.join('.', 'data', dataset)
#
# test_data_r = FB15k_237(path, split='test')[0]
# val_data_r = FB15k_237(path, split='val')[0]
# train_data_r = FB15k_237(path, split='train')[0]
# # print(test_data)
# # print(val_data)
# # print(train_data)
# edge_index = torch.cat((test_data_r.edge_index, val_data_r.edge_index, train_data_r.edge_index), dim=-1)
# graph = Data(edge_index=edge_index, num_nodes=train_data_r.num_nodes)
#
# split = T.RandomLinkSplit(
#     num_val=0.05,
#     num_test=0.1,
#     is_undirected=True,
#     add_negative_train_samples=False,
#     neg_sampling_ratio=1.0,
#  )
#
#
# train_data, val_data, test_data = split(graph)
# print(test_data)
# print(val_data)
# print(train_data)
# Data(edge_index=[2, 488362], num_nodes=14951, edge_label=[54262], edge_label_index=[2, 54262])
# Data(edge_index=[2, 461232], num_nodes=14951, edge_label=[27130], edge_label_index=[2, 27130])
# Data(edge_index=[2, 461232], num_nodes=14951, edge_label=[230616], edge_label_index=[2, 230616])


# Data(x=[2708, 1433], edge_index=[2, 8976], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[4488], edge_label_index=[2, 4488])
# Data(x=[2708, 1433], edge_index=[2, 8976], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[526], edge_label_index=[2, 526])
# Data(x=[2708, 1433], edge_index=[2, 9502], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[1054], edge_label_index=[2, 1054])

dataset = 'JF17k'#FB15k    FB15k-237    JF17k humans_wikidata
print('model: Node2vec \t dataset: ', dataset)
    # osp.dirname(osp.realpath(__file__)), 
path = osp.join('.', 'data', dataset)

val_data = pred_triple(path, split='val')[0]
print(val_data.edge_label_index[0][:64], val_data.edge_type[:64])