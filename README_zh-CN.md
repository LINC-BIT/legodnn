<div align="center">



<img src="https://user-images.githubusercontent.com/20336673/145025177-6dd4d49b-65ed-457d-84a0-d9e716d85039.png" height="100"/>
<img src="https://user-images.githubusercontent.com/73862727/145950351-9924c1e8-64b2-43d6-84d4-255607cb585d.png"/>


![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
![docs](https://img.shields.io/badge/docs-latest-blue)
![license](https://img.shields.io/badge/license-Apache2.0-green)





</div>
 

## 简介
[English](README.md) | 简体中文

 目前使用比较广泛的主要有六种视觉类DNN应用，包括**图像分类、语义分割、目标检测、行为识别、异常检测和姿态估计**。在这六种视觉DNN应用中均包含了大量的卷积层。

![image](https://user-images.githubusercontent.com/73862727/146324643-f0ddfbcc-dfd7-4ef4-b5d3-0e3600e984d0.png)

	
 - **图像分类**是根据各自在图像信息中所反映的不同特征，把不同类别的目标区分开来的图像处理方法。输入为一个图像，通过多个卷积层提取图像的特征，再接全连接层，输出该图像属于某个类别的概率。如图（a）所示的ResNet18。Resnet18可以分为root+四个Stage+全连接层fc，经过ImageNet预训练的Resnet18网络在其他应用中用于提取图像特征，并被成为Backbone。其他应用都是对这四个Stage进行进一步的处理。

- **语义分割**是对图像中的每一个像素点进行分类，目前广泛应用于医学图像和无人驾驶等场景中。语义分割网络通常是一种编码器-解码器结构。如图（b）所示为经典语义分割网络FCN解码器结构。与目标检测网络一样，编码器对应于图像分类网络，用于提取特征，解码器各有不同。

- **目标检测**是功能是检测出图像中目标（如人、狗、车等）对应的检测框的坐标以及目标的识别。主流的目标检测网络可以分为三个部分：Backbone+Neck+检测头。如图（c）所示是Yolov3网络，其Backbone对应图中的Conv和四个Stage，也就是Resnet18中的Root和四个Stage。检测头就是常见的线性预测层。

- **行为识别**是识别出视频片段中目标的行为，如挥手说话等。如图（d）所示为经典的行为识别双流网络。模型被分为空间卷积网络和时间卷积网络，两者都是做分类任务且均使用图像分类网络。
 
 - **异常检测**是检测数据中的异常情况，这里主要是图片和视频的异常检测。异常检测网络主要分为两种：基于Self-training的模型（分类）和基于GAN的模型（重构）。如图（e1）和图（e2）所示。基于Self-traing的模型主要是通过Resnet进行特征提取，全连接层用于分类预测。基于GAN的模型是简单对称的AutoEncoder模型。


	
	
 - **姿态估计**是确定某一三维目标物体的方位指向问题。姿态估计在机器人视觉、动作跟踪等很多领域都有应用。主流的姿态估计网络主要分为两种，第一种是先对图片进行目标检测然后对检测到的单张图片检测关键点，这种网络结构与目标检测类似。第二种是先找出关键点然后对关键点进行分组，从而得到检测结果，这种网络结构与语义分割相似。
	
 
LegoDNN（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)）是一个针对模型缩放问题的轻量级、块粒度、可伸缩的解决方案，根据卷积层从原始DNN模型中抽取块，生成稀疏派生块，然后对这些块进行再训练。通过组合这些块，扩大原始模型的伸缩选项。并且在运行时，通过算法对块的选择进行了优化。以Resnet18为例，如下图所示。 本项目是一个对LegoDNN的基于PyTorch的实现，支持将以上六种主流应用场景下的深度神经网络转换为LegoDNN，从而增加大量的缩放选项，在边缘端进行动态缩放以适应设备资源的变化。
  
  
 <div align="center" padding="10">
   <img src="https://user-images.githubusercontent.com/73862727/145767343-1cddf0f4-a9a9-48ef-8884-57688883e167.png"/>
 </div>
 
  **主要特性**
- **模块化设计**

  本项目将LegoDNN的抽块、再训练等过程解耦成各个模块，通过组合不同的模块组件，用户可以更便捷的对自己的自定义模型Lego化。
  
- **块的自动化抽取**
    
    本项目实现了通用的块的抽取算法（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)），对于图像分类、目标检测、语义分割、姿态估计、行为识别、异常检测等类型的模型均可以通过算法，自动找出其中的块用于再训练。

## 项目整体架构

<div align="center" padding="10">
 <img src="https://user-images.githubusercontent.com/73862727/146190146-32de7e60-1406-4f68-8645-f39854b5dc29.png" />
</div>

**处理流程**主要分为离线阶段和在线阶段。

离线阶段：
- 原始模型通过block extrator抽取出模型中的原始块，然后将这些块通过`decendant block generator`生成稀疏派生块，然后用retrain模块将这些块根据原始数据在原始模型中产生的中间数据进行再训练。最后将原始块以及所有的再生块通过`block profiler`对块进行精度和内存的分析，生成分析文件。

在线阶段：
- 在线阶段首先对离线阶段产生的块进行延迟分析和估计，生成延迟评估文件，然后`scailing optimizer`根据延迟评估文件以及离线阶段生成的块的精度分析文件和内存分析文件在运行时根据算法选择最优的块交给`block swicher`进行切换。


**具体模块说明**
- blockmanager：在本框架中通过blockmanager融合了`block extrator`、`descendant block generator`、`block swicher`的功能，主要负责块的抽取，派生，更换，存储等，本项目已经通过AutoBlockManager实现针对多种模型自动对块的抽取,其算法原理详情见[文章]()。
- **offline**：在离线阶段对块进行再训练以提升其精度，并分析每个块的指标。
  - BlockRetrainer：用于对块的再训练。
  - BlockProfile：用于对块的大小、精度等信息进行分析统计。
- **online**：在线阶段主要是负责分析块与边缘设备相关的指标以及在线运行时针对特定的内存、精度限定对块进行热更新以进行优化。
  - LatencyProfile：用于对块在边缘设备上进行延迟数据的分析。
  - ScailingOptimizer：用于根据特定内存大小对块进行优化热更新。

## 安装
**依赖**
- Linux 和 Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 



**安装流程**
1. 使用conda新建虚拟环境，并进入该虚拟环境
	```
	conda create -n legodnn python=3.6
	conda active legodnn
	```
2. 根据[Pytorch官网](https://github.com/LINC-BIT/IoT-and-Edge-Intelligence)安装Pytorch和torchvision
![image](https://user-images.githubusercontent.com/73862727/146364503-5664de5b-24b1-4a85-b342-3d061cd7563f.png)
根据官网选择要安装的Pytorch对应的参数，然后复制相应的命令在终端输入即可

   **注意请确定安装的是CPU版本Pytorch还是GPU版本，如果是CPU版本的Pytorch请将下面代码中的`device='cuda'`改为`device='cpu'`**
3. 安装legodnn


	```shell
	pip install legodnn
	```
## 开始使用

**离线阶段**
1. 引入组件，初始化随机种子
	```python
	import torch
	from legodnn import BlockRetrainer, BlockProfiler, LagencyEstimator, ScalingOptimizer
	from legodnn.common.utils.dl.common.env import set_random_seed
	set_random_seed(0)
	from legodnn.common.manager.block_manager.auto_block_manager import AutoBlockManager
	from legodnn.common.manager.model_manager.common_model_manager import CommonModelManager
	from cv_task.image_classification.cifar.models import resnet18
	from cv_task.datasets.image_classification.cifar_dataloader import CIFAR100Dataloader
	```
2. 初始化需要处理的模型
	```python
	  teacher_model = resnet18(num_classes=100).to(device)
	  teacher_model.load_state_dict(torch.load('data/model/resnet18/2021-10-20/22-09-22/resnet18.pth')['net'])
	```
3. 通过AutoBlockManager对模型进行自动化的抽取以及生成派生稀疏块，并存储到指定文件夹中
	```python
		cv_task = 'image_classification'
		dataset_name = 'cifar100'
		model_name = 'resnet18'               
		compress_layer_max_ratio = 0.125      # 指定自动化抽取块使，layer的最大ratio
		device = 'cuda'                       # 指定是否使用cuda
		model_input_size = (1, 3, 32, 32)     # 指定模型的输入数据的维度
		block_sparsity = [0.0, 0.3, 0.6, 0.8] # 指定每个块生成多少个派生块以及每个派生快的稀疏度

		root_path = os.path.join('../data/blocks', 
								  cv_task, model_name + '_' 
								  + dataset_name + '_' 
								  + str(compress_layer_max_ratio).replace('.', '-'))
		compressed_blocks_dir_path = root_path + '/compressed'    # 指定存储文件夹
		model_manager = CommonModelManager()
		block_manager = AutoBlockManager(block_sparsity,teacher_model,
										 model_manager,model_input_size,
										 compress_layer_max_ratio,device)
		block_manager.extract_all_blocks(compressed_blocks_dir_path)
	```
4. 对块进行再训练
	```python
	compressed_blocks_dir_path = root_path + '/compressed'   # 指定未训练的块的位置
	trained_blocks_dir_path = root_path + '/trained'         # 指定训练后块的存储位置 
	train_loader, test_loader = CIFAR100Dataloader()         # 指定训练数据和测试数据的loader
	block_training_max_epoch = 20                            # 指定训练过程中的epoch
	block_retrainer = BlockRetrainer(teacher_model, block_manager, model_manager, 
										 compressed_blocks_dir_path,
										 trained_blocks_dir_path, 
										 block_training_max_epoch, 
										 train_loader, 
										 device=device)
	block_retrainer.train_all_blocks()
	```
5. 对块精度和内存大小的分析
	```python
	trained_blocks_dir_path = root_path + '/trained'         # 指定训练后块的存储位置 
	block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
											  trained_blocks_dir_path, test_loader, model_input_size, device)
	block_profiler.profile_all_blocks()
	```

**在线阶段**
1. 对延迟进行计算和估计
	```python
	test_sample_num = 100
	lagency_estimator = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
							   test_sample_num, model_input_size, device)
	lagency_estimator.profile_all_blocks()
	```
2. 在具体的内存大小和推理延迟的条件下选择具体的块来构建模型
	```python
	lagency_estimator = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
								   test_sample_num, model_input_size, device)
	lagency_estimator.profile_all_blocks()
	optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
										   block_manager, model_manager, device)
	optimal_runtime.update_model(10, 4.5 * 1024 ** 2)
	```

**完整的例子**
 
  - 请参考[Demo](example/legodnn_resnet_test.py)


**加入复杂模型**

对于训练方式特殊的模型，需要重新实现`legodnn.common.manager.model_manager.abstract_model_manager`中的`AbstractModelManager`，或者基于`CommonModelManager`进行相关函数的修改
	```python
	class AbstractModelManager(abc.ABC):
		"""Define all attributes of the model.
		"""

		@abc.abstractmethod
		def forward_to_gen_mid_data(self, model: torch.nn.Module, batch_data: Tuple, device: str):
			"""Let model perform an inference on given data.

			Args:
				model (torch.nn.Module): A PyTorch model.
				batch_data (Tuple): A batch of data, typically be `(data, target)`.
				device (str): Typically be 'cpu' or 'cuda'.
			"""
			raise NotImplementedError()

		@abc.abstractmethod
		def dummy_forward_to_gen_mid_data(self, model: torch.nn.Module, model_input_size: Tuple[int], device: str):
			"""Let model perform a dummy inference.

			Args:
				model (torch.nn.Module): A PyTorch model.
				model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
				device (str): Typically be 'cpu' or 'cuda'.
			"""
			raise NotImplementedError()

		@abc.abstractmethod 
		def get_model_acc(self, model: torch.nn.Module, test_loader: DataLoader, device: str):
			"""Get the test accuracy of the model.

			Args:
				model (torch.nn.Module): A PyTorch model.
				test_loader (DataLoader): Test data loader.
				device (str): Typically be 'cpu' or 'cuda'.
			"""
			raise NotImplementedError()

		@abc.abstractmethod
		def get_model_size(self, model: torch.nn.Module):
			"""Get the size of the model file (in byte).

			Args:
				model (torch.nn.Module): A PyTorch model.
			"""
			raise NotImplementedError()

		@abc.abstractmethod
		def get_model_flops_and_param(self, model: torch.nn.Module, model_input_size: Tuple[int]):
			"""Get the FLOPs and the number of parameters of the model, return as (FLOPs, param).

			Args:
				model (torch.nn.Module): A PyTorch model.
				model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
			"""
			raise NotImplementedError()

		@abc.abstractmethod
		def get_model_latency(self, model: torch.nn.Module, sample_num: int, model_input_size: Tuple[int], device: str):
			"""Get the inference latency of the model.

			Args:
				model (torch.nn.Module): A PyTorch model.
				sample_num (int): How many samples is used in the test.
				model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
				device (str): Typically be 'cpu' or 'cuda'.
			"""
			raise NotImplementedError()

	```

## docker（待完善，docker镜像尚未制作完）

使用镜像
**注意目前docker镜像不支持GPU**
|树莓派4B(aarch64)|Jeston TX2(armv8)|
|----|----|
|`docker run -it lincbit/legodnn:raspberry4B-1.0`|`docker run -it lincbit/legodnn:jetsontx2-1.0`|


## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 更新日志

**1.0.0**版本已经在 2021.12.20 发布：

  基础功能实现
  
## 支持的模型

   **图像分类**
  - [x] [ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
  - [x] [MobileNetV2 (CVPR'2018)](https://arxiv.org/abs/1801.04381)
  - [x] [ResNeXt (CVPR'2017)](https://arxiv.org/abs/1611.05431)
 
  **目标检测**
  - [x] [Fast R-CNN (NIPS'2015)](https://ieeexplore.ieee.org/abstract/document/7485869)
  - [x] [YOLOv3 (CVPR'2018)](https://arxiv.org/abs/1804.02767)
  - [x] [FreeAnchor (NeurIPS'2019)](https://arxiv.org/abs/1909.02466)
  
  **语义分割**
  - [x] [FCN (CVPR'2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)
  - [X] [U-Net (MICCAI'2016)](https://arxiv.org/abs/1505.04597)
  - [x] [DeepLab v3 (ArXiv'2017)](https://arxiv.org/abs/1706.05587)

  
  **异常检测**
  - [x] [GANomaly (ACCV'2018)](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)
  - [x] [GPND (NIPS'2018)](https://arxiv.org/abs/1807.02588)
  - [x] [Self-Training (CVPR'2020)](Self-trainedDeepOrdinalRegressionforEnd-to-EndVideoAnomalyDetection)
  
  **姿态估计**
  - [x] [DeepPose (CVPR'2014)](https://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html)
  - [x] [SimpleBaselines2D (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)
    
  **行为识别**
  - [x] [TSN (ECCV'2016)](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)
  - [x] [TRN (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)
  
  
  
