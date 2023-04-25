from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset


class pred_triple(InMemoryDataset):
    r"""
        ../data/dataset
        train.txt: rel src dst
        val.txt: rel src dst lab
        train.txt: rel src dst lab
    """
    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        raise Exception(f"no such file or dictionary {self.raw_paths[0]}")
        
    def process(self):
        data_list, node_dict, rel_dict = [], {}, {}
        # edge_index = torch.empty(0, dtype=torch.long)
        for path in self.raw_paths:
            split = path.split('\\')[-1].split('.')[0]
            with open(path, 'r') as f:
                data = [x.split(' ') for x in f.read().split('\n')[:-1]]
                
            edge_label = torch.ones(len(data), dtype=torch.long)
            edge_label_index = torch.empty((2, len(data)), dtype=torch.long)
            edge_type = torch.empty(len(data), dtype=torch.long)
            if split == 'train':
               
                for i, (rel, src, dst) in enumerate(data):
                    if src not in node_dict:
                        node_dict[src] = len(node_dict)
                    if dst not in node_dict:
                        node_dict[dst] = len(node_dict)
                    if rel not in rel_dict:
                        rel_dict[rel] = len(rel_dict)
                        # print(rel, len(rel_dict))
    
                    edge_label_index[0, i] = node_dict[src]
                    edge_label_index[1, i] = node_dict[dst]
                    edge_type[i] = rel_dict[rel]
                
                edge_label = torch.ones(len(data), dtype=torch.long)
                # edge_index = torch.cat((edge_index, edge_label_index), dim=1)
            else:

                for i, (rel, src, dst, lab) in enumerate(data):
                    if src not in node_dict:
                        node_dict[src] = len(node_dict)
                    if dst not in node_dict:
                        node_dict[dst] = len(node_dict)
                    if rel not in rel_dict:
                        rel_dict[rel] = len(rel_dict)
    
                    edge_label[i] = int(lab)
                    edge_label_index[0, i] = node_dict[src]
                    edge_label_index[1, i] = node_dict[dst]
                    edge_type[i] = rel_dict[rel]
                # mm = [True if l else False for l in edge_label.tolist()]
                # edge_index = torch.cat((edge_index, edge_label_index[[[mm, mm]]].reshape(2, -1)), dim=1)
            

            data = Data(edge_type=edge_type, edge_label=edge_label, edge_label_index=edge_label_index )
            data_list.append(data)
    
        for data, path in zip(data_list, self.processed_paths):
            # data.edge_index = edge_index
            data.num_nodes = len(node_dict)
            torch.save(self.collate([data]), path)
