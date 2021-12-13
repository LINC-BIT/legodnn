<div align="center">


<img src="https://user-images.githubusercontent.com/20336673/145025177-6dd4d49b-65ed-457d-84a0-d9e716d85039.png" width="375"/>

![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
![docs](https://img.shields.io/badge/docs-latest-blue)
![license](https://img.shields.io/badge/license-Apache2.0-green)

![图片3](https://user-images.githubusercontent.com/73862727/145766996-4fc31c70-317d-42a6-a3fc-2f825d7dde28.png)



</div>
 

## 简介

  LegoDNN（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)）是一个针对模型缩放问题的轻量级、块粒度、可伸缩的解决方案。本项目是一个对LegoDNN的基于PyTorch的实现。
  
  **主要特性**
- **模块化设计**

  本项目将LegoDNN的抽块、再训练等过程解耦成各个模块，通过组合不同的模块组件，用户可以更便捷的对自己的自定义模型Lego化。
  
- **块的自动化抽取**
    
    本项目实现了通用的块的抽取算法（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)），对于图像分类、目标检测、语义分割、姿态估计、行为识别、异常检测等类型的模型均可以通过算法，自动找出其中的块用于再训练。

**项目整体架构**
<div align="left" padding="10">
 <img src="https://user-images.githubusercontent.com/20336673/145038154-e698821f-5d42-4457-a4cb-9bf412959365.png" width="500"/>
</div>

- **common**：主要是对块的管理以及模型的管理，以便于离线阶段和在线阶段的使用。
  - modelmanager：主要负责中间数据的生成以及获取模型的总精度和总延迟。对于不同的模型，这些功能的实现或有些许不一样，需要基于AbstractModelManager来针对不同的模型进行实现。
  - blockmanager：主要负责块的抽取，更换，存储等，本项目已经通过AutoBlockManager实现针对多种模型自动对块的抽取,其算法原理详情见[文章]()。
- **offline**：在离线阶段对块进行再训练以提升其精度，并分析每个块的指标。
  - retrain：属于离线阶段对块的再训练。
  - Profile：属于离线阶段对块的大小、精度等信息进行分析统计。
- **online**：在线阶段主要是负责分析块与边缘设备相关的指标以及在线运行时针对特定的内存、精度限定对块进行热更新以进行优化。
  - LatencyProfile：属于在线阶段对块在边缘设备上进行延迟数据的分析。
  - RuntimeOptimizer：属于在线阶段运行期间根据特定内存大小对块进行热更新。

## 安装

在安装legodnn之前，请确保Pytorch已经成功安装在环境中，可以参考PyTorch的官方[文档](https://pytorch.org/)

```shell
pip install legodnn
```
简单的例子
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, '../../')
sys.setrecursionlimit(100000)
import torch
from legodnn import BlockExtractor, BlockTrainer, ServerBlockProfiler, EdgeBlockProfiler, OptimalRuntime
from legodnn.gen_series_legodnn_models import gen_series_legodnn_models
from legodnn.utils.dl.common.env import set_random_seed
set_random_seed(0)
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_model_manager import CommonModelManager

from cv_task.datasets.image_classification.cifar_dataloader import CIFAR10Dataloader, CIFAR100Dataloader
from cv_task.image_classification.cifar.models import resnet18

if __name__ == '__main__':
    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'resnet18'
    # compress_layer_max_ratio = 0.25
    compress_layer_max_ratio = 0.125
    device = 'cuda' 
    model_input_size = (1, 3, 32, 32)
    train_batch_size = 128
    test_batch_size = 128
    block_sparsity = [0.0, 0.3, 0.6, 0.8]
    root_path = os.path.join('results/legodnn', 
                             cv_task, model_name+'_'
                             +dataset_name + '_' 
                             + str(compress_layer_max_ratio).replace('.', '-'))
    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    block_training_max_epoch = 20
    test_sample_num = 100
    
    teacher_model = resnet18(num_classes=100).to(device)
    teacher_model.load_state_dict(
            torch.load('cv_task_model/image_classification/cifar100/resnet18/2021-10-20/22-09-22/resnet18.pth')['net'])

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
    block_extractor = BlockExtractor(teacher_model, block_manager, compressed_blocks_dir_path, model_input_size, device)
    block_extractor.extract_all_blocks()

    print('\033[1;36m-------------------------------->    START BLOCK TRAIN\033[0m')
    train_loader, test_loader = CIFAR100Dataloader()
    block_trainer = BlockTrainer(teacher_model, block_manager, model_manager, compressed_blocks_dir_path,
                                 trained_blocks_dir_path, block_training_max_epoch, train_loader, device=device)
    block_trainer.train_all_blocks()

    server_block_profiler = ServerBlockProfiler(teacher_model, block_manager, model_manager,
                                                trained_blocks_dir_path, test_loader, model_input_size, device)
    server_block_profiler.profile_all_blocks()


    edge_block_profiler = EdgeBlockProfiler(block_manager, model_manager, trained_blocks_dir_path, 
                                            test_sample_num, model_input_size, device)
    edge_block_profiler.profile_all_blocks()

    optimal_runtime = OptimalRuntime(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    gen_series_legodnn_models(deadline=100, model_size_search_range=[15,50], target_model_num=50, 
                              optimal_runtime=optimal_runtime, 
                              descendant_models_save_path=descendant_models_dir_path, 
                              device=device)

```

## docker

使用镜像

|树莓派4B|Jeston|
|----|----|
|`docker run -it lincbit/legodnn:raspberry4B-1.0`|`docker run -it lincbit/legodnn:jetsontx2-1.0`|


## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 更新日志

**1.0.0**版本已经在 2021.12.20 发布：

  基础功能实现
  
## 基准测试和支持的模型

  **图像分类**
  - [x] VGG (ICLR'2015)
  - [x] InceptionV3 (CVPR'2016)
  - [x] ResNet (CVPR'2016)
  - [x] CBAM (ECCV'2018)
 
  **目标检测**
  - [x] Fast R-CNN (NIPS'2015)
  - [x] YOLOv3 (CVPR'2018)
  - [x] CenterNet (ICCV'2019)
  
  **语义分割**
  - [x] FCN (CVPR'2015)
  - [X] SegNet (TPAMI'2017)
  - [x] DeepLab v3 (ArXiv'2017)
  - [x] CCNet (ICCV'2021)
  
  **姿态估计**
  - [x] DeepPose (CVPR'2014)
  - [x] CPN (CVPR'2018)
  - [x] SimpleBaselines (ECCV'2018)
    
  **行为识别**
  - [x] Two-STeam CNN (NIPS'2014)
  - [x] TSN (ECCV'2016)
  - [x] TRN (ECCV'2018)
  
  **异常检测**
  - [x] GANomaly (ACCV'2018)
  - [x] GPND (NIPS'2018)
  - [x] OGNet (CVPR'2020)
  - [x] Self-Training (CVPR'2020)
  
