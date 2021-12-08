import sys
import time

import torch
from PIL import Image
from torchvision import transforms

# sys.path.insert(0, '../../')

from legodnn.common.manager import CommonModelManager,CommonBlockManager
from optimal_runtime import OptimalRuntime
from legodnn.common.utils import set_random_seed
from legodnn.common.dataloader import CIFAR10Dataloader
from legodnn.common.model import ResNet110_cifar10

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


if __name__ == '__main__':
    set_random_seed(0)

    # basic config
    device = 'cpu'
    # teacher_model = ResNet110_cifar10(pretrained=True, device=device)
    train_loader, test_loader = CIFAR10Dataloader('../data/datasets')
    model_input_size = (1, 3, 32, 32)
    
    # legodnn config
    block_sparsity = [0.0, 0.125, 0.25, 0.375, 0.5]
    compressed_blocks_dir_path = '../data/blocks/resnet110/resnet110-cifar10-m6/compressed'
    trained_blocks_dir_path = '../data/blocks/resnet110/resnet110-cifar10-m6/trained'
    block_training_max_epoch = 20
    test_sample_num = 100
    
    default_blocks_id = ResNet110BlockManager.get_default_blocks_id()
    model_manager = CommonModelManager()
    block_manager = ResNet110BlockManager(default_blocks_id,
                                          [block_sparsity for _ in range(len(default_blocks_id))], model_manager)
    
    # pipeline start
    # block_extractor = BlockExtractor(teacher_model, block_manager, compressed_blocks_dir_path, model_input_size, device)
    # block_extractor.extract_all_blocks()

    # block_trainer = BlockTrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path, trained_blocks_dir_path,
    #                              block_training_max_epoch, train_loader, device=device)
    # block_trainer.train_all_blocks()
    #
    # server_block_profiler = ServerBlockProfiler(teacher_model, block_manager, model_manager, trained_blocks_dir_path,
    #                                             test_loader, model_input_size, device)
    # server_block_profiler.profile_all_blocks()
    
    # edge_block_profiler = EdgeBlockProfiler(block_manager, model_manager, trained_blocks_dir_path, test_sample_num,
    #                                         model_input_size, device)
    # edge_block_profiler.profile_all_blocks()
    
    optimal_runtime = OptimalRuntime(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    start_time = time.time()
    print('optimal model info: {}'.format(optimal_runtime.update_model(10, 4.5 * 1024 ** 2)))
    end_time = time.time()
    duration  = end_time - start_time
    print(duration)
    #
    model = optimal_runtime._pure_runtime.get_model()
    # print(model)
    model.eval()
    # for i, (data,target) in enumerate(train_loader):
    #     # data = data.to(device, dtype=data.dtype, non_blocking=False, copy=False),
    #     # print(data.shape)
    #     # print(data)
    #
    #     with torch.no_grad():
    #         print(data.shape)
    #         start_time = time.time()
    #         output = model(data[0:1])
    #         end_time = time.time()
    #         print(end_time - start_time)
    #         output = [torch.argmax(out).item() for out in output]
    #
    #         print(torch.tensor(output))
    #         # print(output.shape)
    #         print(target)
    #     # print(data.shape)
    #     break

    import threading
    def func():

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ])
        img_pil = Image.open('../data/test_data/airplane.jpg')  # PIL.Image.Image对象
        img_pil = transform_train(img_pil)

        img_pil2 = Image.open('../data/test_data/airplane2.jpg')  # PIL.Image.Image对象
        img_pil2 = transform_train(img_pil2)

        img_pil3 = Image.open('../data/test_data/cat.jpg')  # PIL.Image.Image对象
        img_pil3 = transform_train(img_pil3)

        img_pil4 = Image.open('../data/test_data/cat2.jpg')  # PIL.Image.Image对象
        img_pil4 = transform_train(img_pil4)

        img_pil5 = Image.open('../data/test_data/horse.jpg')  # PIL.Image.Image对象
        img_pil5 = transform_train(img_pil5)
        data = torch.stack([img_pil,img_pil2,img_pil3,img_pil4,img_pil5],dim=0)
        result = model(data)
        print([torch.argmax(result,1)])
    threading.Thread(target=func).start()
    # from matplotlib import pyplot
    # from scipy.misc import toimage
    # for i, (data,target) in enumerate(train_loader):
    # # create a grid of 3x3 images
    # # for i in range(0, 9):
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(toimage(data[i]))
    #     if i == 9:
    #         break
    # # show the plot
    # pyplot.show()
