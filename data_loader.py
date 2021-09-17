from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import LastFMAsia
from torch_geometric.datasets import GitHub
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import AmazonProducts
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

class Dataset(Data):
    def __init__(self, data_name, data_path):
        super().__init__()

        if data_name in ['Cora', 'Citeseer', 'PubMed']:
            # Cora: node_num=2708, edge_num=5278, feature_num=1433, class_num=7
            # Citeseer: node_num=3327, edge_num=4552, feature_num=3703, class_num=6
            # PubMed: node_num=19717, edge_num=44324, feature_num=500, class_num=3
            self.dataset = Planetoid(root=data_path, name=data_name) # can use transform=T.NormalizeFeatures()
        elif data_name == 'FacebookPagePage': 
            # node_num=22470, edge_num=171002, feature_num=128, class_num=4
            self.dataset = FacebookPagePage(root=data_path+'/'+data_name)
        elif data_name == 'LastFMAsia':  
            # node_num=7624, edge_num=27806, feature_num=128, class_num=18
            self.dataset = LastFMAsia(root=data_path+'/'+data_name)
        elif data_name == 'GitHub':  
            # node_num=37700, edge_num=289003, feature_num=128, class_num=2
            self.dataset = GitHub(root=data_path+'/'+data_name)
        elif data_name == 'Actor':  
            # node_num=7600, edge_num=15009, feature_num=932, class_num=5
            self.dataset = Actor(root=data_path+'/'+data_name)
        elif data_name in ['Cornell', 'Texas', 'Wisconsin']:
            self.dataset = WebKB(root=data_path, name=data_name)
        elif data_name == 'WikiCS':  
            # node_num=11701, edge_num=148555, feature_num=300, class_num=10
            self.dataset = WikiCS(root=data_path+'/'+data_name)
        elif data_name == 'AmazonProducts':  
            # OOM issue: node_num=1569960, edge_num=132169734, feature_num=200, class_num=107
            self.dataset = AmazonProducts(root=data_path+'/'+data_name)
        elif data_name == 'Yelp':  
            # OOM issue: node_num=716847, edge_num=6977409, feature_num=300, class_num=100
            self.dataset = Yelp(root=data_path+'/'+data_name)
        elif data_name == 'Flickr':  
            # node_num=89250, edge_num=449878, feature_num=500, class_num=7
            self.dataset = Flickr(root=data_path+'/'+data_name)
        elif data_name == 'Reddit':  
            # OOM Issue: node_num=232965, edge_num=57307946, feature_num=602, class_num=41
            self.dataset = Reddit(root=data_path+'/'+data_name)
        elif data_name == 'Reddit2':  
            # OOM Issue: node_num=232965, edge_num=11606919, feature_num=602, class_num=41
            self.dataset = Reddit2(root=data_path+'/'+data_name)
        elif data_name in ['Computers', 'Photo']: 
            # Computers: node_num=13752, edge_num=245861, feature_num=767, class_num=10
            # Photo: node_num=7650, edge_num=119081, feature_num=745, class_num=8
            self.dataset = Amazon(root=data_path, name=data_name)
        elif data_name in ['CS', 'Physics']: 
            # CS: node_num=18333, edge_num=81894, feature_num=6805, class_num=15
            # Physics: node_num=34493, edge_num=247962, feature_num=8415, class_num=5
            self.dataset = Coauthor(root=data_path, name=data_name)
        elif data_name in ['Cora_Full', 'Cora_ML', 'Citeseer_Full', 'DBLP']: 
            # Cora_Full: node_num=19793, edge_num=63421, feature_num=8710, class_num=70
            # Cora_ML: node_num=2995, edge_num=8158, feature_num=2879, class_num=7
            # Citeseer_Full: node_num=4230, edge_num=5337, feature_num=602, class_num=6
            # DBLP: node_num=17716, edge_num=52867, feature_num=1639, class_num=4
            if data_name == 'Cora_Full':
                data_name = 'Cora'
            if data_name == 'Citeseer_Full':
                data_name = 'Citeseer'
            self.dataset = CitationFull(root=data_path, name=data_name)
        else:
            self.dataset = None
            exit('unknown dataset: {}'.format(data_name))        

        train_ratio = 0.5
        val_ratio = 0.2
        test_ratio = 0.3
        data, train_mask, val_mask, test_mask = self.split(train_ratio, val_ratio, test_ratio)
        
        self.x = data.x
        self.y = data.y
        self.edge_index = data.edge_index

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.num_nodes = data.num_nodes
        self.num_node_features = data.num_node_features
        self.num_edges = data.num_edges
        self.num_classes = self.dataset.num_classes

        print('=== Data statistics ===')
        log = 'Node num: {}\nNode feature size: {}\nNode class num: {}\nEdge num: {}'
        print(log.format(self.num_nodes, self.num_node_features, self.num_classes, int(self.num_edges/2)))
        log = 'Train:val:test = {}:{}:{} = {}:{}:{}\n'
        print(log.format(train_ratio, val_ratio, test_ratio, torch.sum(self.train_mask), torch.sum(self.val_mask), torch.sum(self.test_mask)))


    def split(self, train_ratio, val_ratio, test_ratio):
        data = self.dataset.get(0)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        train_mask.fill_(False)
        for c in range(self.dataset.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            num_train_per_class = int(idx.size(0) * train_ratio)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask.fill_(False)
        num_val = int(remaining.size(0) * (val_ratio/(val_ratio + test_ratio)))
        val_mask[remaining[:num_val]] = True

        test_mask.fill_(False)
        num_test = int(remaining.size(0) * (test_ratio/(val_ratio + test_ratio)))
        test_mask[remaining[num_val:num_val + num_test]] = True

        return data, train_mask, val_mask, test_mask


# Download and process data at './dataset/ogbg_molhiv/'
# dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
# dataset = PygNodePropPredDataset(name = d_name) 

 
# split_idx = dataset.get_idx_split() 
# train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
# valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
# test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
# train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# graph = dataset[0] # pyg graph object
