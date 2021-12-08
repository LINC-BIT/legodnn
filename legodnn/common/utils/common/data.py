import numpy as np


def min_max_normalize(data):
    data = np.asarray(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))
