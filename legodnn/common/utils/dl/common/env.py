import random
import torch,os
import numpy as np


def set_random_seed(seed: int):
    """Fix all random seeds in common Python packages (`random`, `torch`, `numpy`). 
    Recommend to use before all codes to ensure reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True