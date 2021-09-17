import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from utils import get_gcn_model, get_hyperparameter
from data_loader import Dataset


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model', type=str, default='GCN', help='graph embedding model')
    parser.add_argument('--dataset', type=str, default='FacebookPagePage', help='dataset')
    parser.add_argument('--num_eval', type=int, default=20, help='number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--data_path', type=str, default='data', help='path to save data')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_eval_pool = np.arange(0, args.epoch_eval_train+1, 10).tolist()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    torch.manual_seed(1)  # Set the random seed so things involved torch.randn are repetable 
    

    dataset = Dataset(args.dataset, args.data_path)
    
    model = get_gcn_model(args.model, dataset).to(args.device)
    lr, weight_decay = get_hyperparameter(args.model)
    dataset = dataset.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def test():
        model.eval()
        pred, accs = model(dataset.x, dataset.edge_index).argmax(dim=1), []
        for _, mask in dataset('train_mask', 'val_mask', 'test_mask'):
            acc  = pred[mask].eq(dataset.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    best_val_acc = test_acc = 0
    for epoch in range(1, args.epoch_eval_train+1):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if epoch in model_eval_pool:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        
if __name__ == '__main__':
    main()