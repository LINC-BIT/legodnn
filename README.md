<div align="center">



<img src="https://user-images.githubusercontent.com/20336673/145025177-6dd4d49b-65ed-457d-84a0-d9e716d85039.png" height="100"/>
<img src="https://user-images.githubusercontent.com/73862727/145950351-9924c1e8-64b2-43d6-84d4-255607cb585d.png"/>


![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
![docs](https://img.shields.io/badge/docs-latest-blue)
![license](https://img.shields.io/badge/license-Apache2.0-green)





</div>
 

## Introduction
English | [简体中文](README_zh-CN.md)

 At present, there are six kinds of visual DNN applications widely used, including image classification, semantic segmentation, object detection, action recognition, anomaly detection and pose estimation.  The six visual DNN applications all contain a large number of convolution layers.  

![image](https://user-images.githubusercontent.com/73862727/146324643-f0ddfbcc-dfd7-4ef4-b5d3-0e3600e984d0.png)

	
 - **Image classification** is an image processing method to distinguish different categories of object from an image. The method first takes an image as input, then extracts the image's feature via convolutional layers, and finally outputs the probability of categories via fully connected layers. Take ResNet18 as example, which is shown in Figure (a), it can be divided into three parts: root, four stages and fully connected layer. Other applications use Resent18 pre-trained by ImageNet to extract image features, and make further modifications on the four stages. The pre-trained ResNet18 is so called Backbone. 

- **Semantic segmentation** is aimed to classify pixels of an image, and has been widely used in the medical image field and unmanned vehicles field. A semantic segmentation network is usually based on an encoder-decoder structure; Figure (b) shows a classical FCN decoder structure. The same as the object detection network, encoder is used to extract features of images while decoder's usage is various.

- **Object detection** is used to detect coordinates of the frames containing objects (e.g., people, dogs, cars) and recognize the objects. Its mainstream networks can be divided into three parts: Backbone, net and detector. Figure (c) shows a popular object detection network YOLO-V3. Its backbone consists of Conv and four stages, that is, the root and four stages in ResNet18; its detector is a linearly prediction layer.

- **Action recognition**  can recognize an object's actions in video clips, such as speaking, waving, etc. As shown in Figure (d), a classical two-stream convolutional networks for action recognition in videos is presented. The network is divided into spatial convolutional network and temporal convolutional network, both of which use image classification networks to perform classification tasks.
 
 - **Anomaly detection**  is used to detect anomalies in data, particularly the image and video data. This network can be divided into two categories: (1) self-training-based model; (2) GAN-based model. As shown in Figure (e1) and Figure (e2), self-training-based model uses ResNet18 to extracts data's feature, uses fully connected layer to make prediction; GAN-based model is a simple and symmetric AutoEncoder model.

	
	
 - **Pose estimation** focuses on the problem of identifying the orientation of a 3-D object. It has been widely used in many fields such as robot vision, motion tracking, etc. The mainstream pose estimation networks are mainly divided into two categories. The first one first detects an object from an image, and then detects the key points of the object. Network structure of this category is similar to objection detection's. In contrast, the second one first finds the key points and then groups the points. In this way, it can obtain the detect results. Network structure of the second one is similar to semantic segmentation's.
	
 
LegoDNN（[Paper](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)）is a lightweight, block-grained and scalable solution for running multi-DNN wrokloads in mobile vision systems. It extracts the blocks of original models via convolutional layers, generates sparse blocks, and retrains the sparse blocks. By composing these blocks, LegoDNN expands the scaling options of original models. At runtime, it optimizes the block selection process using optimization algorithms. The following figure shows a LegoDNN example of ResNet18. This project is a PyTorch-based implementation of LegoDNN, and allow to convert the deep neural networks in the above six mainstream applications to LegoDNN. With the LegoDNN, original models are able to dynamically scale at the edge, and adapt to the change of device resources.
  


 <div align="center" padding="10">
   <img src="https://user-images.githubusercontent.com/73862727/146643884-fcb3f56a-c4d3-457c-9b6e-b930a2720d5c.png"/>
 </div>
 
  **Major features**
- **Modular Design**

  This project decomposes  the block extracting,retraining and selecting processes of legodnn into various modules. Users can  conver their own custom model to legodnn more conveniently by using these module components.  
  
- **Automatic extraction of blocks**
    
    This project has implemented a general block extraction algorithm, supporting the automatic block extraction of the models in image classification, target detection, semantic segmentation, attitude estimation, behavior recognition, anomaly detection applications.

## Architecture

<div align="center" padding="10">
 <img src="https://user-images.githubusercontent.com/73862727/146190146-32de7e60-1406-4f68-8645-f39854b5dc29.png" />
</div>

**Architecture of legodnn** is split into offline stage and online stage.

Offline Stage：
- At offline stage, the `block extrator` extracts the raw blocks from orginal model,and feeds them to the `decendant block generator` to generate descendant block.Then `block retrainer` retrain the descendant blocks.Finally,`block profiler` profile all blocks' informations about accuracy and memory.

Online Stage：
- At online stage, the `latency estimator` estimate the latency of each block at edge device,then pass the estimate latency with the information about accuracy and memory together to the `scaling optimater` to select blocks optimally.Finally,`block swicher` replaces the corresponding blocks in the model with the selected blocks at runtime.


**Module details**
- **BlockManager**: this module integrates `block extractor`, `descendant block generator`, and `block switcher`. The block extractor is responsible for extracting original blocks from an original model's convolution layers. The descendant block generator is responsible for pruning the original blocks to generate multiple sparsity descendant blocks. The block switcher is responsible for replacing blocks with optimal blocks at run time, where the optimal blocks are selected by optimization algorithms. With the AutoBlockManager, this project has implemented automatic extraction of blocks for various models.
- **BlockRetrainer**：this module is used to retrain descendant models to inprove their accuracies. The retraining takes the intermediate data as training data and the sparse blocks as models; the intermediate data is generated by original models as well as original training data; the sparse blocks are generated by original models. The retraining process is quite fast because it only used the intermediate data, reducing the model computation. Meanwhile, these intermediate data can be used in parallel to train the descendant blocks generated from the same original blocks.
- **BlockProfile**：this module is used to generate analysis and statistics information of the block size, accuracy, etc. The size of a block is the memory it occupies. Since the accuracy loss of a block is different in different combined models, this module selects k different size combined models that contains the block to calculate.
- **LatencyProfile**：this module is used to analyze the latency reduction percentage of the blocks in edge devices. The inference latency is obtained by simulating each block's inference in edge device directly. The latency reduction percentage of each block is calculated by using the following formula: (latency of the original block - latency of the currently derived block)/latency of the original block.
- **ScailingOptimizer**：this module is used to update and optimize the blocks in real time. By formalizing the block selection as an integer linear programming optimization problem and resolving it in a real time, we can continuously obtain the model that owns the maximal accuracy and satisfies the conditions of specific latency and memory limitation.

## Installation
**Prerequisites**
- Linux 和 Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 



**Prepare environment**
1. Create a conda virtual environment and activate it.
	```
	conda create -n legodnn python=3.6
	conda active legodnn
	```
2. Install PyTorch and torchvision according the [official site](https://github.com/LINC-BIT/IoT-and-Edge-Intelligence)
![image](https://user-images.githubusercontent.com/73862727/146364503-5664de5b-24b1-4a85-b342-3d061cd7563f.png)
Get install params according to the selection in the official site,and copy them to the terminal.

   **Note: please determine whether the CPU version of pytorch or GPU version is installed. If the CPU version of pytorch is installed, please change the `device ='cuda'`in the following code to `device ='cpu'`**
3.  Install legodnn


	```shell
	pip install legodnn
	```
## Getting Started

**Offline stage**
1. Import components and initialize seed feed
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
2. Initialize orginal model
	```python
	  teacher_model = resnet18(num_classes=100).to(device)
	  teacher_model.load_state_dict(torch.load('data/model/resnet18/2021-10-20/22-09-22/resnet18.pth')['net'])
	```
3. Extract the blocks automatically ,then generate descendant blocks and save the blocks to disk using AutoBlockManager
	```python
		cv_task = 'image_classification'
		dataset_name = 'cifar100'
		model_name = 'resnet18'               
		compress_layer_max_ratio = 0.125      
		device = 'cuda'                       
		model_input_size = (1, 3, 32, 32)     
		block_sparsity = [0.0, 0.3, 0.6, 0.8] 

		root_path = os.path.join('../data/blocks', 
								  cv_task, model_name + '_' 
								  + dataset_name + '_' 
								  + str(compress_layer_max_ratio).replace('.', '-'))
		compressed_blocks_dir_path = root_path + '/compressed'    
		model_manager = CommonModelManager()
		block_manager = AutoBlockManager(block_sparsity,teacher_model,
										 model_manager,model_input_size,
										 compress_layer_max_ratio,device)
		block_manager.extract_all_blocks(compressed_blocks_dir_path)
	```
4. Retrain the blocks
	```python
	compressed_blocks_dir_path = root_path + '/compressed'   
	trained_blocks_dir_path = root_path + '/trained'         
	train_loader, test_loader = CIFAR100Dataloader()         
	block_training_max_epoch = 20                            
	block_retrainer = BlockRetrainer(teacher_model, block_manager, model_manager, 
										 compressed_blocks_dir_path,
										 trained_blocks_dir_path, 
										 block_training_max_epoch, 
										 train_loader, 
										 device=device)
	block_retrainer.train_all_blocks()
	```
5. Get the profiles about accuracy and memory of the blocks.
	```python
	trained_blocks_dir_path = root_path + '/trained'         # 指定训练后块的存储位置 
	block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
											  trained_blocks_dir_path, test_loader, model_input_size, device)
	block_profiler.profile_all_blocks()
	```

**Online stage**
1. Estimate latency time of the block 
	```python
	test_sample_num = 100
	lagency_estimator = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
					     test_sample_num, model_input_size, device)
	lagency_estimator.profile_all_blocks()
	```
2. Select the blocks optimally
	```python
	lagency_estimator = LagencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
								   test_sample_num, model_input_size, device)
	lagency_estimator.profile_all_blocks()
	optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
					   block_manager, model_manager, device)
	optimal_runtime.update_model(10, 4.5 * 1024 ** 2)
	```

**Full example**
 
  - Please refer to [Demo](example/legodnn_resnet_test.py)


**Complex Model**

The model have particular training need to impletment a custom model manager based on  AbstractModelManager in package `legodnn.common.manager.model_manager.abstract_model_manager`.

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

## docker（Docker image is not finished yet）

Using docker
**Note that these Docker images do not support GPU**
|Raspberry pi 4B(aarch64)|Jeston TX2(armv8)|
|----|----|
|`docker run -it lincbit/legodnn:raspberry4B-1.0`|`docker run -it lincbit/legodnn:jetsontx2-1.0`|


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

**1.0.0** was released in 2021.12.20：

  Implement basic functions
  
## Supported models 

  **Image classfication**
  - [x] [ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
  - [x] [MobileNetV2 (CVPR'2018)](https://arxiv.org/abs/1801.04381)
  - [x] [ResNeXt (CVPR'2017)](https://arxiv.org/abs/1611.05431)
 
  **Obejct detection**
  - [x] [Fast R-CNN (NIPS'2015)](https://ieeexplore.ieee.org/abstract/document/7485869)
  - [x] [YOLOv3 (CVPR'2018)](https://arxiv.org/abs/1804.02767)
  - [x] [FreeAnchor (NeurIPS'2019)](https://arxiv.org/abs/1909.02466)
  
  **Semantic segmentation**
  - [x] [FCN (CVPR'2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)
  - [X] [U-Net (MICCAI'2016)](https://arxiv.org/abs/1505.04597)
  - [x] [DeepLab v3 (ArXiv'2017)](https://arxiv.org/abs/1706.05587)

  
  **Anomaly detection**
  - [x] [GANomaly (ACCV'2018)](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)
  - [x] [GPND (NIPS'2018)](https://arxiv.org/abs/1807.02588)
  - [x] [Self-Training (CVPR'2020)](Self-trainedDeepOrdinalRegressionforEnd-to-EndVideoAnomalyDetection)
  
  **Pose estimation**
  - [x] [DeepPose (CVPR'2014)](https://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html)
  - [x] [SimpleBaselines2D (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)
    
  **Action recognition**
  - [x] [TSN (ECCV'2016)](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)
  - [x] [TRN (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)
  
