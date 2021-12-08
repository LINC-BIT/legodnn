import os

from .config import config

data_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
__all__ = ["common","offline","online","config","data_path"]
# print(data_path)