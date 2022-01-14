import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.setrecursionlimit(100000)
import torch
import copy
from legodnn import BlockRetrainer, BlockProfiler, LagencyEstimator, ScalingOptimizer
from legodnn.common.utils.dl.common.model import get_model_size, set_module
from legodnn.common.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.common.detection.model_topology_extraction import topology_extraction
from legodnn.common.manager.block_manager.auto_block_manager import AutoBlockManager
from legodnn.common.detection.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.common.utils.gen_series_legodnn_models import gen_series_legodnn_models

from legodnn.common.manager.model_manager.common_action_recognition_model_manager_v2 import CommonActionRecognitionModelManager
from cv_task.datasets.action_recognition import mmaction_build_dataloader
from cv_task.action_recognition.mmaction_models.legodnn_configs import get_trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_config
from cv_task.action_recognition.mmaction_tools import mmaction_init_model
from mmcv.parallel import MMDataParallel

if __name__=='__main__':
    cv_task = 'action_recognition'
    dataset_name = 'sthv2'
    model_name = 'mmaction_trn_r50_1x1x8_50e_sthv2_rgb'
    # compress_layer_max_ratio = 0.25
    compress_layer_max_ratio = 0.125
    device = 'cuda'
    model_input_size = (1, 8, 3, 256, 256)
    # model_input_size = (8,3,256,256)
    train_batch_size = 8
    test_batch_size = 1
    block_sparsity = [0.0, 0.3, 0.6, 0.8]
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    block_training_max_epoch = 20
    test_sample_num = 100
    model_config = get_trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_config()

    teacher_pt_file = None
    checkpoint = None

    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    jit_detector = mmaction_init_model(model_config, None, mode='lego_jit', device=device)
    model_graph = topology_extraction(jit_detector, model_input_size, device=device)
    model_graph.print_ordered_node()
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()
    
    model_manager = CommonActionRecognitionModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)
    
    if teacher_pt_file is not None:
        teacher_detector = mmaction_init_model(config=model_config, checkpoint=None, mode='mmdet_test', device=device)
        raw_teacher = torch.load(teacher_pt_file).to(device)
        for name, module in raw_teacher.named_modules():
            if len(list(module.children()))>0:
                continue
            else:
                set_module(teacher_detector, name, copy.deepcopy(module))
    else:
        teacher_detector = mmaction_init_model(config=model_config, checkpoint=checkpoint, mode='mmdet_test', device=device)
    
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_manager.extract_all_blocks(teacher_detector, compressed_blocks_dir_path, model_input_size, device)
    
    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = mmaction_build_dataloader(cfg=model_config)
    parallel_teacher_detector = MMDataParallel(teacher_detector.cuda(0), device_ids=[0])
    block_trainer = BlockRetrainer(parallel_teacher_detector, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    block_trainer.train_all_blocks()
    # exit(0)
    server_block_profiler = BlockProfiler(teacher_detector, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()

    edge_block_profiler = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path, 
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()
    # exit(0)
    optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    model_size_min = get_model_size(torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt')))/1024**2
    model_size_max = get_model_size(teacher_detector)/1024**2 + 1
    gen_series_legodnn_models(deadline=100, model_size_search_range=[model_size_min, model_size_max], target_model_num=50, optimal_runtime=optimal_runtime, descendant_models_save_path=descendant_models_dir_path, device=device)
