from legodnn.common.manager import CommonBlockManager
import sys
import yaml
from legodnn.common.third_party.nni_new.algorithms.compression.pytorch.pruning import *

class Config:
    def __init__(self,config_file_path:str):
        with open(config_file_path,"r") as f:
            file_data = f.read()
            self.data = yaml.load(file_data,yaml.CLoader)

    def get_value(self,name:str):
        if "." in name:
            names = name.split(".")
            value = self.data
            for name in names:
                value = value.get(name,"")
            return value
        else:
            return self.data.get(name,"")

    def get_block_manager(self):
        return self._get_class_from_value("block_manager",
                                          "legodnn.common.manager",
                                          CommonBlockManager)

    def get_pruner(self):
        return self._get_class_from_value("pruner_class_name",
                                          "legodnn.common.third_party.nni_new.algorithms.compression.pytorch.pruning",
                                          L1FilterPruner)

    def _get_class_from_value(self,name:str,package:str,default_class):
        value = self.data.get(name)
        if not value:
            return default_class
        else:
            return getattr(sys.modules.get(package),value)


config = Config("../config.yaml")