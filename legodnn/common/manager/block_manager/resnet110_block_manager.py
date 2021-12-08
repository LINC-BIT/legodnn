from .common_block_manager import CommonBlockManager
out_info = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
class ResNet110BlockManager(CommonBlockManager):
    @staticmethod
    def get_default_blocks_id():
        all_basic_blocks_name = []
        for i in range(3):
            for j in range(18):
                all_basic_blocks_name += ['layer{}.{}'.format(i + 1, j)]

        # merge 6 BasicBlocks into a LegoDNN block
        factor = 6
        res = []
        for i in range(len(all_basic_blocks_name)):
            if i % factor == 0:
                j = i + factor if i < len(all_basic_blocks_name) - factor else len(all_basic_blocks_name)
                res += ['|'.join(all_basic_blocks_name[i: j])]

        return res