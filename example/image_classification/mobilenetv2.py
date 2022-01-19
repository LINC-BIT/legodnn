import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.setrecursionlimit(100000)
import torch
from legodnn import BlockRetrainer, BlockProfiler, LatencyEstimator, ScalingOptimizer
from legodnn.common.utils.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.common.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.common.detection.model_topology_extraction import topology_extraction
from legodnn.common.manager.block_manager.auto_block_manager import AutoBlockManager
from legodnn.common.detection.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.common.manager.model_manager.common_model_manager import CommonModelManager
from legodnn.common.utils.dl.common.model import get_model_size

from cv_task.datasets.image_classification.cifar_dataloader import CIFAR10Dataloader, CIFAR100Dataloader
from cv_task.image_classification.cifar.models import mobilenetv2_w2

if __name__ == '__main__':
    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'mobilenetv2_w2'
    compress_layer_max_ratio = 0.125
    device = 'cuda' 
    model_input_size = (1, 3, 32, 32)
    train_batch_size = 128
    test_batch_size = 128
    block_sparsity = [0.0, 0.3, 0.6, 0.8]
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    block_training_max_epoch = 20
    test_sample_num = 100
    
    teacher_model = mobilenetv2_w2(num_classes=100).to(device)
    teacher_model.load_state_dict(torch.load('your pre-trained weight'))

    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    model_graph = topology_extraction(teacher_model, model_input_size, device=device)
    model_graph.print_ordered_node()
    # exit(0)
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio) # resnet18
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()
    # exit(0)
    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)

    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_manager.extract_all_blocks(teacher_model, compressed_blocks_dir_path, model_input_size, device)

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = CIFAR100Dataloader()
    block_trainer = BlockRetrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    block_trainer.train_all_blocks()

    server_block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()
    
    edge_block_profiler = LatencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()
    
    optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    model_size_min = get_model_size(torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt')))/1024**2
    model_size_max = get_model_size(teacher_model)/1024**2 + 1
    gen_series_legodnn_models(deadline=100, model_size_search_range=[model_size_min, model_size_max], target_model_num=50, optimal_runtime=optimal_runtime, descendant_models_save_path=descendant_models_dir_path, device=device)
