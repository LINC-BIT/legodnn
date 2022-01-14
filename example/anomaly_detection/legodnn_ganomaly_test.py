import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, '../../')
sys.setrecursionlimit(100000)
import torch
from legodnn import BlockRetrainer, BlockProfiler, LagencyEstimator, ScalingOptimizer
from legodnn.common.utils.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.common.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.common.detection.model_topology_extraction import topology_extraction
from legodnn.common.manager.block_manager.auto_block_manager import AutoBlockManager
from legodnn.common.detection.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.common.manager.model_manager.common_anomaly_detection_model_manager import CommonAnomalyDetectionModelManager

from cv_task.datasets.anomaly_detection.ganomaly_coil100 import ganomaly_coil100_dataloader
from cv_task.anomaly_detection.models.ganomaly import ganomaly_coil100_netg

if __name__ == '__main__':
    cv_task = 'anomaly_detection'
    dataset_name = 'coil100'
    model_name = 'ganomaly'
    compress_layer_max_ratio = 0.125
    device = 'cuda' 
    model_input_size = (1, 3, 128, 128)
    train_batch_size = 128
    test_batch_size = 128
    block_sparsity = [0.0, 0.3, 0.6, 0.8]
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    block_training_max_epoch = 20
    test_sample_num = 100
    
    teacher_model = ganomaly_coil100_netg(pretrained=True, device=device)
    
    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    model_graph = topology_extraction(teacher_model, model_input_size, device=device)
    model_graph.print_ordered_node()
    # exit(0)
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()
    # exit(0)
    model_manager = CommonAnomalyDetectionModelManager(model_name=model_name)
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)
    
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_extractor = BlockExtractor(teacher_model, block_manager, compressed_blocks_dir_path, model_input_size, device)
    block_extractor.extract_all_blocks()

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = ganomaly_coil100_dataloader()
    block_trainer = BlockRetrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    block_trainer.train_all_blocks()
    # exit(0)
    server_block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()


    edge_block_profiler = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path, 
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()

    optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    gen_series_legodnn_models(deadline=100, model_size_search_range=[0,12], target_model_num=50, optimal_runtime=optimal_runtime, descendant_models_save_path=descendant_models_dir_path, device=device)
