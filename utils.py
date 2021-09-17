import torch
from gcn_models import GCN, Sage, SGC, GAT, GAT2, Transformer


def get_gcn_model(model_name, dataset):
    if model_name == 'GCN':
        model = GCN(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'Sage':
        model = Sage(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'SGC':
        model = SGC(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'GAT':
        model = GAT(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'GAT2':
        model = GAT2(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'Transformer':
        model = Transformer(dataset.num_node_features, dataset.num_classes)
    else:
        model = None 
        exit('unknown model: {}'.format(model_name))

    return model


def get_hyperparameter(model_name):
    if model_name == 'GCN':
        lr = 0.01
        weight_decay = 5e-4
    elif model_name == 'Sage':
        lr = 0.01
        weight_decay = 5e-4
    elif model_name == 'SGC':
        lr = 0.2
        weight_decay  = 0.005
    elif model_name == 'GAT':
        lr = 0.005
        weight_decay = 5e-4
    elif model_name == 'GAT2':
        lr = 0.005
        weight_decay = 5e-4
    elif model_name == 'Transformer':
        lr = 0.005
        weight_decay = 5e-4

    return lr, weight_decay

# def initialize_syn_graph():
