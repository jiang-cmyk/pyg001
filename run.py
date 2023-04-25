from sklearn.metrics import roc_auc_score, f1_score

import numpy as np,  json, argparse, time
from pprint import pprint
import logging.config
import os.path as osp
import sys

import torch

from dataload import pred_triple

from torch_geometric.nn import Node2Vec


class Runner(object):

    def load_data(self):
        
        path = osp.join('.', 'data', self.p.dataset)
        self.train_data = pred_triple(path, split='train')[0]
        self.val_data = pred_triple(path, split='val')[0]
        self.test_data = pred_triple(path, split='test')[0]
        
        max_index = torch.max(self.train_data['edge_label_index'])
        
        mm = [False if self.val_data.edge_label_index[0][ind] > max_index  
              or self.val_data.edge_label_index[1][ind] > max_index else True 
              for ind in range(len(self.val_data.edge_label_index[0]))]
        
        nn = [False if self.test_data.edge_label_index[0][ind] > max_index 
               or self.test_data.edge_label_index[1][ind] > max_index else True 
               for ind in range(len(self.test_data.edge_label_index[0]))]
        
        self.val_data.edge_index = torch.cat((self.val_data.edge_label_index[0][mm], self.val_data.edge_label_index[1][mm])).reshape(2, -1)
        self.val_data.edge_label = self.val_data.edge_label[mm]
        
        self.test_data.edge_index = torch.cat((self.test_data.edge_label_index[0][nn], self.test_data.edge_label_index[1][nn])).reshape(2, -1)
        self.test_data.edge_label = self.test_data.edge_label[nn]
        
        
        
    def get_logger(self, name, log_dir, config_dir):

        config_dict = json.load(open( config_dir + 'log_config.json'))
        config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-').replace(':', '-')

        logging.config.dictConfig(config_dict)
        logger = logging.getLogger(name)

        std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(std_out_format))
        logger.addHandler(consoleHandler)

        return logger


    def __init__(self, params):
        
        self.p            = params
        self.logger        = self.get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
  
        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = self.add_model(self.p.model)
        self.optimizer    = self.add_optimizer()
        self.loader       = self.get_loader()


    def add_model(self, model):
        
        if   model.lower()    == 'node2vec':     
            model = Node2Vec(self.train_data.edge_label_index,
                        # torch.cat((train_data.edge_label_index, test_data.edge_label_index), dim=1), 
                            embedding_dim=self.p.embedding_dim, walk_length=self.p.walk_length,
                            context_size=self.p.context_size, walks_per_node=self.p.walks_per_node,
                            num_negative_samples=self.p.num_negative_samples, p=self.p.p, q=self.p.q, sparse=True)
  
        else: raise NotImplementedError

        model.to(self.device)
        return model


    def add_optimizer(self):

        return torch.optim.SparseAdam(list(self.model.parameters()), lr=self.p.lr)


    def get_loader(self):
        num_workers = 0 if sys.platform.startswith('win') else 4
        return self.model.loader(batch_size=self.p.batch_size, shuffle=True, num_workers=num_workers)


    def save_model(self, save_path):

        state = {
            'state_dict'    : self.model.state_dict(),
            'optimizer'    : self.optimizer.state_dict(),
            'args'        : vars(self.p)
        }
        torch.save(state, save_path)


    def load_model(self, load_path):
        state            = torch.load(load_path)
        state_dict        = state['state_dict']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])


    def decode(self, z, edge_label_index):
        # print(edge_label_index.shape)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
 

    def predict(self, data):

        self.model.eval()

        with torch.no_grad():
            z = self.model()
            # mm = [True if data.edge_label_index[0][ind]< len(z)  and data.edge_label_index[1][ind] <len(z) else False for ind in range(len(data.edge_label_index[0]))]
            out = self.decode(z, data.edge_index).view(-1).sigmoid()
            y_pred = np.around(out.cpu().numpy(),0).astype(int) 
        # print()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()), f1_score(data.edge_label.cpu().numpy(), y_pred)



    def run_epoch(self):
        
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)


    def fit(self):
        
        self.logger.info('train_data: ')
        self.logger.info(self.train_data)
        self.logger.info('test data: ')
        self.logger.info(self.test_data)
        self.logger.info('val data: ')
        self.logger.info(self.val_data)
        
        for epoch in range(self.p.max_epochs):
            loss = self.run_epoch()
            val_auc, val_f1 = self.predict(self.val_data)
            self.logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
    
        test_auc, test_f1= self.predict(self.test_data)
        self.logger.info(f'Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}')
        
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',        default='testrun',                    help='Set run name for saving/restoring models')
    parser.add_argument('-data',        dest='dataset',         default='JF17k',            
                        help='Dataset to use, select from{FB15k, FB15k-237, JF17k, humans_wikidata}, default: FB15k-237 ')
    parser.add_argument('-model',        dest='model',        default='Node2vec',        help='Model Name')

    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gpu',        type=str,               default='0',            help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',        dest='max_epochs',     type=int,       default=100,      help='Number of epochs')
    # parser.add_argument('-l2',        type=float,             default=0.0,            help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',        type=float,             default=0.01,            help='Starting Learning Rate')
    parser.add_argument('-embedding_dim',   dest='embedding_dim',      default=128,    type=int,       help='embedding dim for node2vc')
    parser.add_argument('-walk_length',   dest='walk_length',      default=20,    type=int,       help='walk length for node2vc')
    parser.add_argument('-context_size',   dest='context_size',      default=10,    type=int,       help='walks_per_node for node2vc')
    parser.add_argument('-walks_per_node',   dest='walks_per_node',      default=10,    type=int,       help='context size for node2vc')
    parser.add_argument('-num_negative_samples',   dest='num_negative_samples',      default=1,    type=int,       help='num negative samples for node2vc')
    parser.add_argument('-p',   dest='p',      default=1,    type=int,       help='p for node2vc')
    parser.add_argument('-q',   dest='q',      default=1,    type=int,       help='q for node2vc')
    parser.add_argument('-num_workers',    type=int,               default=2,                     help='Number of processes to construct batches')

    parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    model = Runner(args)
    model.fit()