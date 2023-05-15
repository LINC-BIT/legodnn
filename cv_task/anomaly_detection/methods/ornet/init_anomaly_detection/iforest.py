import torch
from sklearn.ensemble import IsolationForest
from typing import List
import numpy as np


_cache = {}


def iforest(datasets: List[torch.Tensor], tested_sample: torch.Tensor, cache_iforest=None, seed=0):
    datasets = np.array([d.cpu().detach().numpy() for d in datasets])
    tested_sample = tested_sample.cpu().detach().numpy()

    if cache_iforest is None:
        sk_iforest = IsolationForest(n_estimators=100,
                                    max_samples=256,
                                    random_state=seed)
        sk_iforest.fit(datasets)
        cache_iforest = sk_iforest
    else:
        sk_iforest = cache_iforest
    
    score, = sk_iforest.decision_function([tested_sample])
    
    return score, sk_iforest
