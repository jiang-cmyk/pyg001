#
# a = [True, False, True, False]
# b = [1, 3, 4, 5]
# print(b[a])
import torch
import os.path as osp
import sys
from FB15k_237 import  FB15k_237
# x = torch.tensor([1, 2, 3, 2, 3, 4, 5])
# print(x[:int(len(x)/3)])
# import sys
# print(sys.version)
# m1 = [1,2,3,4]
# m2 = [2,3,4,5]
# m=[m1,m2]
# n = [[1 for _ in range(len(m[0]))] for __ in range(len(m))]
#
# n[1][:]=m[1][:]
# print([1]*10)
# dataset = 'FB15k_237'
#     # osp.dirname(osp.realpath(__file__)), 
# path = osp.join('..', 'data', dataset)
#
# train_data = FB15k_237(path, split='val')[0]
# print(train_data)
#
#
# # print(train_data['edge_index'][0].max())
# # print(train_data['edge_index'][1].max())
# # print(train_data['edge_type'].max())
#
a = torch.tensor([1,2,3], dtype=float)
b = torch.tensor([[0.8,0.1,0.1],
                  [0.2,0.4,0.4],
                  [0.8,0.1,0.1]])
c = torch.arange(25).reshape((5, -1))
# print(c)
d = [True, False, True, False]
# print(torch.where(a>2, a, 5))
# # a = torch.cat((a, b), dim=1)
# c = torch.empty(0)
# c = torch.cat((c, b), dim=1)
# c = torch.cat((c, a), dim=1)
# m =torch.tensor([0, 1, 1, 0])
# n = torch.tensor([1,2,3,4])
# mm = [True if l else False for l in m.tolist()]

# print(a[[[mm, mm]]].reshape(2, -1))
# print(m.device)
# from sklearn.metrics import   f1_score
# y_ =torch.tensor([0, 1, 1, 1, 1, 1, 0, 0, 1, 1])
# y = torch.tensor( [2.2775894e-05, 5.5642217e-01, 9.9827433e-01, 9.0191436e-01,
#                     9.3406904e-01, 9.3491459e-01, 9.4044596e-01, 9.9720204e-01, 9.8131895e-01, 9.9641490e-01])
# # print(y.numpy())
# f1_score(y_, y, average='macro')

from sklearn.metrics import roc_auc_score
#
print(roc_auc_score(a, b, multi_class='ovr'))
#
# a = [1,0,0,1]
# b = [0,1,2,1]
# print(a.mean(dim=1).shape)
# print(roc_auc_score(a, b))
# print(torch.cat((a, b)).reshape(2, -1).max())

#
# from torch_geometric.nn import Node2Vec
# edge_label_index = torch.tensor([
#     [1,2,3,4,5,6,7,8],
#     [2,3,4,5,6,7,8,9]
#     ])
# model = Node2Vec(edge_label_index,
#                     # torch.cat((train_data.edge_label_index, test_data.edge_label_index), dim=1), 
#                 embedding_dim=128, walk_length=20,
#                 context_size=10, walks_per_node=10,
#                 num_negative_samples=1, p=1, q=1, sparse=True)
#
# optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
#
# save_path = './log/mm'
# print(torch.load(save_path))


# x = torch.tensor([5,3,5,7,
#                   4,3,65,45,
#                   8,5,3,21,
#                   23,5,67678,3]).reshape(4, -1)
# print(torch.argsort(x, dim=1, descending=True))
# print(torch.argsort(torch.argsort(x, dim=1, descending=True), dim=1, descending=False))
# from torch.nn import functional as F
#
# print(torch.softmax(b, dim=1))
# print(torch.sigmoid(b))