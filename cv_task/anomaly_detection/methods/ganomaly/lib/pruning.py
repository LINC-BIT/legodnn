import torch
import copy

from cv_task.anomaly_detection.methods.ganomaly.lib.networks import NetG
from utils.pruning import prune_module
from utils.model import get_module_convs_name


def prune_netg(netg: NetG, nc, isize, zsize, sparsity):
    pruned_netg = copy.deepcopy(netg)
    for m in ['encoder1', 'decoder', 'encoder2']:
        model = getattr(pruned_netg, m)
        input_size = (1, nc, isize, isize) if 'encoder' in m else (1, zsize, 1, 1)
        pruned_model = prune_module(model, get_module_convs_name(model, 1), sparsity, input_size, 'cuda')
        setattr(pruned_netg, m, pruned_model)
    
    return pruned_netg
