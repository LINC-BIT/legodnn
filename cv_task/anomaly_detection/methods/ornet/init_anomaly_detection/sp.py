import torch
from sklearn.utils import resample
from typing import List


# E-distance between two frames
def _euclidean_d(a: torch.Tensor, b: torch.Tensor):
    return torch.norm(a - b, 2).item()


def sp(datasets: List[torch.Tensor], tested_sample: torch.Tensor, sample_num=20, bootstrap_num=20, seed=0):
    avg_min_d = 0.
    
    for new_seed in range(bootstrap_num):
        new_seed += seed
        
        min_d = 1e8
        samples = resample(datasets, n_samples=sample_num, random_state=new_seed)
        
        for sample in samples:
            min_d = min(_euclidean_d(tested_sample, sample), min_d)
            
        avg_min_d += min_d
        
    return avg_min_d / bootstrap_num
