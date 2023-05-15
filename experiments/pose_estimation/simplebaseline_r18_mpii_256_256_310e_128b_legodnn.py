import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import sys
sys.path.insert(0, '../../../')
sys.setrecursionlimit(100000)
from legodnn import BlockExtractor, BlockTrainer, ServerBlockProfiler, EdgeBlockProfiler, OptimalRuntime
from legodnn.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_pose_estimation_model_manager_v2 import CommonPoseEstimationModelManager
from legodnn.utils.common.file import experiments_model_file_path
from mmcv.parallel import MMDataParallel
from legodnn.utils.dl.common.model import get_module, set_module, get_model_size

from cv_task.datasets.pose_estimation import mmpose_build_dataloader
from cv_task.pose_estimation.mmpose_models.legodnn_configs import get_simplebaseline_res18_mpii_256_256_310e_128b_config
from cv_task.pose_estimation.mmpose_tools import mmpose_init_model, train_posenet_by_config

if __name__=='__main__':
    cv_task = 'pose_estimation'
    dataset_name = 'mpii'
    model_name = 'mmpose_simplebaseline_res18_mpii'
    method = 'legodnn'
    compress_layer_max_ratio = 0.125
    epoch_num = 62
    device = 'cuda' 
    model_input_size = (1, 3, 256, 256)
    block_sparsity = [0.0, 0.2, 0.4, 0.6, 0.8]

    teacher_pt_file = None
    checkpoint = None
    
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    test_sample_num = 100
    model_config = get_simplebaseline_res18_mpii_256_256_310e_128b_config()
    
    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    jit_detector = mmpose_init_model(model_config, None, mode='lego_jit', device=device)
    model_graph = topology_extraction(jit_detector, model_input_size, device=device, mode='unpack')
    model_graph.print_ordered_node()
    
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()

    model_manager = CommonPoseEstimationModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)
    teacher_detector = mmpose_init_model(config=model_config, checkpoint=checkpoint, mode='mmpose_test', device=device)
    
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_extractor = BlockExtractor(teacher_detector, block_manager, compressed_blocks_dir_path, model_input_size, device)
    block_extractor.extract_all_blocks()

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = mmpose_build_dataloader(cfg=model_config)
    parallel_teacher_detector = MMDataParallel(teacher_detector.cuda(0), device_ids=[0])
    block_trainer = BlockTrainer(parallel_teacher_detector, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, epoch_num, train_loader, device=device)
    block_trainer.train_all_blocks()

    server_block_profiler = ServerBlockProfiler(teacher_detector, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()

    edge_block_profiler = EdgeBlockProfiler(block_manager, model_manager, trained_blocks_dir_path, 
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()

    optimal_runtime = OptimalRuntime(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    model_size_min = get_model_size(torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt')))/1024**2
    model_size_max = get_model_size(teacher_detector)/1024**2 + 1
    gen_series_legodnn_models(deadline=100, model_size_search_range=[model_size_min, model_size_max], target_model_num=100, optimal_runtime=optimal_runtime, descendant_models_save_path=descendant_models_dir_path, device=device)

