import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, r'.')
sys.setrecursionlimit(100000)

import torch
from legodnn import BlockRetrainer, BlockProfiler, LatencyEstimator, ScalingOptimizer
from legodnn.common.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.common.manager.block_manager.auto_block_manager import AutoBlockManager
from legodnn.common.manager.model_manager.common_model_manager import CommonModelManager
from cv_task.image_classification.cifar.models import resnet18
from cv_task.datasets.image_classification.cifar_dataloader import CIFAR100Dataloader
if __name__ == '__main__':
    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'resnet18'
    compress_layer_max_ratio = 0.125
    device = 'cuda'
    model_input_size = (1, 3, 32, 32)
    block_sparsity = [0.0, 0.3, 0.6, 0.8]

    root_path = os.path.join("data","blocks", cv_task, model_name + '_' + dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'



    
    teacher_model = resnet18(num_classes=100).to(device)
    teacher_model.load_state_dict(torch.load('data/model/resnet18/2021-10-20/22-09-22/resnet18.pth')['net'])

    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity,teacher_model,model_manager,model_input_size,compress_layer_max_ratio,device)
    
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_manager.extract_all_blocks(compressed_blocks_dir_path)

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = CIFAR100Dataloader()




    block_training_max_epoch = 20
    # block_retrainer = BlockRetrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path,
    #                                  trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    # block_retrainer.train_all_blocks()
    
    # block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
    #                                       trained_blocks_dir_path, test_loader, model_input_size, device)
    # block_profiler.profile_all_blocks()
    
    test_sample_num = 100
    latency_estimator = LatencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
                               test_sample_num, model_input_size, device)
    latency_estimator.profile_all_blocks()
    
    optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
                                       block_manager, model_manager, device)
    optimal_runtime.update_model(10, 45 * 1024 ** 2)
