import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import torch
import math
from cv_task.semantic_segmentation.mmseg_tools.test import test_segmentor
import sys
sys.path.insert(0, '../../')
sys.setrecursionlimit(100000)
from legodnn import BlockExtractor, BlockTrainer, ServerBlockProfiler, EdgeBlockProfiler, OptimalRuntime
from legodnn.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_semantic_segmentation_model_manager_v2 import CommonSemanticSegmentationModelManager
from legodnn.utils.dl.common.model import get_module, set_module, get_model_size
from legodnn.utils.common.file import experiments_model_file_path

from cv_task.datasets.semantic_segmentation import mmseg_build_dataloader
from cv_task.semantic_segmentation.mmseg_models.legodnn_configs import get_fcn_r18_d8_512x512_b16_30k_voc2012_config
from cv_task.semantic_segmentation.mmseg_tools import mmseg_init_model
from mmcv.parallel import MMDataParallel

if __name__=='__main__':
    cv_task = 'semantic_segmentation'
    dataset_name = 'voc2012'
    model_name = 'fcn_r18_d8_512_512_b16'
    method = 'legodnn'
    sparsity = 's0'
    compress_layer_max_ratio = 0.125
    # compress_layer_max_ratio = 0.25
    # train_iter_num = 6000
    train_iter_num = 1000
    device = 'cuda' 
    model_input_size = (1, 3, 512, 512)
    block_sparsity = [0.0, 0.4, 0.8]
    teacher_pt_file = None
    checkpoint = None
    
    root_path = os.path.join('results/legodnn', cv_task, model_name+'_'+dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))
    
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    test_sample_num = 65
    model_config = get_fcn_r18_d8_512x512_b16_30k_voc2012_config()
    
    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    jit_detector = mmseg_init_model(model_config, None, mode='lego_jit', device=device)
    model_graph = topology_extraction(jit_detector, model_input_size, device=device, mode='unpack')
    model_graph.print_ordered_node()
    
    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()
    # exit(0)
    model_manager = CommonSemanticSegmentationModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)    
    # TODO
    if teacher_pt_file is not None:
        teacher_detector = mmseg_init_model(config=model_config, checkpoint=None, mode='mmseg_test', device=device)
        raw_teacher: torch.nn.Module = torch.load(teacher_pt_file).to(device)
        for name,param in raw_teacher.named_parameters():
            param.requires_grad = True
            
        for name, module in raw_teacher.named_modules():
            if len(list(module.children()))>0:
                continue
            else:
                set_module(teacher_detector, name, copy.deepcopy(module))
    else:
        teacher_detector = mmseg_init_model(config=model_config, checkpoint=checkpoint, mode='mmseg_test', device=device)
        
    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    block_extractor = BlockExtractor(teacher_detector, block_manager, compressed_blocks_dir_path, model_input_size, device)
    block_extractor.extract_all_blocks()
    # exit(0)
    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = mmseg_build_dataloader(cfg=model_config)
    epoch_num = math.ceil(float(train_iter_num)/len(train_loader))
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