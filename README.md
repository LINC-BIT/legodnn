<div align="center">



<img src="https://user-images.githubusercontent.com/20336673/145025177-6dd4d49b-65ed-457d-84a0-d9e716d85039.png" height="100"/>
<img src="https://user-images.githubusercontent.com/73862727/145950351-9924c1e8-64b2-43d6-84d4-255607cb585d.png"/>


![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
![docs](https://img.shields.io/badge/docs-latest-blue)
![license](https://img.shields.io/badge/license-Apache2.0-green)





</div>

## Table of contents
- [1 Introduction](#1-introduction)
  * [1.1 Major features](#11-major-features)
  * [1.2 Architecture](#12-architecture)
- [2 Code and Installation](#2-code-and-installation)
  * [2.1 Code](#21-code)
  * [2.2 Installation](#22-installation)
- [3 Repository of DNNs in vision tasks](#3-repository-of-dnns-in-vision-tasks)
  * [3.1 Supported models](#31-supported-models)
  * [3.2 Full examples](#32-full-examples)
  * [3.3 How to implement new models in LegoDNN](#33-how-to-implement-new-models-in-legodnn)
- [4 Demo Video and experiment data](#4-demo-video-and-experiment-data)

## 1 Introduction

 At present, LegoDNN (https://dl.acm.org/doi/abs/10.1145/3447993.3483249) includes six kinds of widely used visual DNN applications, including image classification, semantic segmentation, object detection, action recognition, anomaly detection and pose estimation. The DNNs in all visual applications contain a large number of convolution layers and blocks.  

![image](https://user-images.githubusercontent.com/73862727/146324643-f0ddfbcc-dfd7-4ef4-b5d3-0e3600e984d0.png)

	
 - **Image classification** applications distinguish different categories of object from an image. The method first takes an image as input, then extracts the image's feature via convolutional layers, and finally outputs the probability of categories via fully connected layers. Take ResNet18 as example, which is shown in Figure (a), it can be divided into three parts: root, four stages and fully connected layer. Other applications use Resent18 pre-trained by ImageNet to extract image features, and make further modifications on the four stages. The pre-trained ResNet18 is so called Backbone. 

- **Semantic segmentation** applications are widely used in medical images and driverless scenes. A typical DNN model has an  encoder-decoder structure, in which the encoder corresponds to an image classification network and the decoder varies across different DNNs. For example, in fully convolutional networks (FCN)~\cite{long2015fully} (Figure (b)), the encoder corresponds to the four stages in ResNet and the decoder contains four convolution layers.

- **Object detection** applications detect coordinates of the frames containing objects (e.g., people, dogs, cars) and recognize the objects. Its mainstream networks can be divided into three parts: Backbone, net and detector. Figure (c) shows a popular object detection network YOLO-V3. Its backbone is a ResNet18 which is divided into two parts :a root convolution layer and four stages here. Its detector is the two conected convolution layers before each output. All the remaining convolution layers form the net.  

- **Action recognition** applications recognize an object's actions in video clips, such as speaking, waving, etc. As shown in Figure (d), a classical two-stream convolutional networks is presented. The network is divided into spatial convolutional network and temporal convolutional network, both of which use image classification networks to perform classification tasks.
 
 - **Anomaly detection** applications detect anomalies in data, particularly the image and video data. This network can be divided into two categories: (1) self-training-based model; (2) GAN-based model. As shown in Figure (e1) and Figure (e2), self-training-based model uses ResNet18 to extract data's feature, use fully connected layer to make prediction; GAN-based model is a simple and symmetric AutoEncoder model.

	
 - **Pose estimation** focuses on the problem of identifying the orientation of a 3-D object. It has been widely used in many fields such as robot vision, motion tracking, etc. The mainstream pose estimation networks are mainly divided into two categories. The first one first detects an object from an image, and then detects the key points of the object. Network structure of this category is similar to objection detection's. In contrast, the second one first finds the key points and then groups the points. In this way, it can obtain the detect results. Network structure of the second one is similar to semantic segmentation's.
	
 
LegoDNN([Paper](https://dl.acm.org/doi/abs/10.1145/3447993.3483249))is a lightweight, block-grained and scalable solution for running multi-DNN wrokloads in mobile vision systems. It extracts the blocks of original models via convolutional layers, generates sparse blocks, and retrains the sparse blocks. By composing these blocks, LegoDNN expands the scaling options of original models. At runtime, it optimizes the block selection process using optimization algorithms. The following figure shows a LegoDNN example of ResNet18. This project is a PyTorch-based implementation of LegoDNN, and allows to convert the deep neural networks in the above six mainstream applications to LegoDNN. With  LegoDNN, original models are able to dynamically scale at edge, and adapt to changing device resources.
  


 <div align="center" padding="10">
	
  

   <img src="https://user-images.githubusercontent.com/73862727/149609666-602d961e-ff28-4042-9492-5e4b19774f2d.png"/>
 </div>
 
 ### 1.1 Major features
- **Modular Design**

  This project decomposes the block extracting, retraining and selecting processes of legodnn into various modules. Users can convert their own custom model to legodnn more conveniently by using these module components.  
  
- **Automatic extraction of blocks**
    
    This project has implemented a general block extraction algorithm, supporting the automatic block extraction of the models in image classification, target detection, semantic segmentation, attitude estimation, behavior recognition, and anomaly detection applications.

### 1.2 Architecture

<div align="center" padding="10">
 <img src="https://user-images.githubusercontent.com/73862727/146190146-32de7e60-1406-4f68-8645-f39854b5dc29.png" />
</div>

 **Architecture of legodnn** is split into the offline stage and the online stage.

- Offline Stage：
	- At the offline stage, the`block extrator`identifies the original/uncompressed blocks from a DNN model, and feeds them to the `decendant block generator` module to produce descendant blocks. The `block retrainer` module then retrains the descendant blocks. Finally, the `block profiler` module profiles all blocks' accuracies, memory and latency information.

- Online Stage：
	- At the online stage, the `latency estimator` module estimates latencies of the blocks at edge devices, then sends these latencies with the accuracy and memory information together to the `scaling optimater` module to optimally select blocks. Finally, the `block swicher` module replaces the corresponding blocks in the model with the selected blocks at runtime.


**Module details**
- **BlockManager**: this module integrates `block extractor`, `descendant block generator`, and `block switcher`. The block extractor is responsible for extracting original blocks from an original model's convolution layers. The descendant block generator is responsible for pruning the original blocks to generate multiple sparsity descendant blocks. The block switcher is responsible for replacing blocks with optimal blocks at run time, where the optimal blocks are selected by optimization algorithms. With the AutoBlockManager, this project has implemented automatic extraction of blocks for various models.
- **BlockRetrainer**：this module is used to retrain descendant models to inprove their accuracies. The retraining takes the intermediate data as training data and the sparse blocks as models; the intermediate data is generated by original models as well as original training data; the sparse blocks are generated by original models. The retraining process is quite fast because it only used the intermediate data, reducing the model computation. Meanwhile, these intermediate data can be used in parallel to train the descendant blocks generated from the same original blocks.
- **BlockProfile**：this module is used to generate analysis and statistics information of the block size, accuracy, etc. The size of a block is the memory it occupies. Since the accuracy loss of a block is different in different combined models, this module selects k different sizes in profiling.
- **LatencyProfile**：this module is used to analyze the latency reduction percentage of the blocks on edge devices. The inference latency is obtained by simulating each block's inference on edge device directly. The latency reduction percentage of each block is calculated by using the following formula: (latency of the original block - latency of the currently derived block)/latency of the original block.
- **ScailingOptimizer**：this module is used to update and optimize the blocks in real time. By formalizing the block selection as an integer linear programming optimization problem and resolving it in real time, we can continuously obtain the model that owns the maximal accuracy and satisfies the conditions of specific latency and memory limitation.

## 2 Code and Installation
### 2.1 Code
**Offline stage**
1. Import components and initialize seed 
	```python
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

	from cv_task.datasets.image_classification.cifar_dataloader import CIFAR10Dataloader, CIFAR100Dataloader
	from cv_task.image_classification.cifar.models import resnet18
	```
2. Initialize original model
	```python
	teacher_model = resnet18(num_classes=100).to(device)
    teacher_model.load_state_dict(torch.load('data/model/resnet110/resnet18.pth')['net'])
	```
3. Extract the blocks automatically, then generate descendant blocks and save the blocks to disk using AutoBlockManager
	```python
	cv_task = 'image_classification'
	dataset_name = 'cifar100'
	model_name = 'resnet18'
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

	model_graph = topology_extraction(teacher_model, model_input_size, device=device)
	model_graph.print_ordered_node()

	detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio) # resnet18
	detection_manager.detection_all_blocks()
	detection_manager.print_all_blocks()

	model_manager = CommonModelManager()
	block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)
	block_manager.extract_all_blocks(teacher_model, compressed_blocks_dir_path, model_input_size, device)

	```
4. Retrain the blocks
	```python
     
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
	trained_blocks_dir_path = root_path + '/trained'         
	block_profiler = BlockProfiler(teacher_model, block_manager, model_manager,
											  trained_blocks_dir_path, test_loader, model_input_size, device)
	block_profiler.profile_all_blocks()
	```

**Online stage**
1. Estimate latency of the block 
	```python
	test_sample_num = 100
	latency_estimator = LatencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
					     test_sample_num, model_input_size, device)
	latency_estimator.profile_all_blocks()
	```
2. Select the blocks optimally
	```python
	latency_estimator = LatencyEstimator(block_manager, model_manager, trained_blocks_dir_path,
								   test_sample_num, model_input_size, device)
	latency_estimator.profile_all_blocks()
	optimal_runtime = ScalingOptimizer(trained_blocks_dir_path, model_input_size,
					   block_manager, model_manager, device)
	optimal_runtime.update_model(10, 4.5 * 1024 ** 2)
	```
	
### 2.2 Installation
**Prerequisites**
- Linux and Windows 
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
	git clone https://github.com/LINC-BIT/legodnn.git
	pip install -r requirements.txt
	```

4. Docker
   
	 Using docker
	**Note that these Docker images do not support GPU**
	|Raspberry pi 4B or Jeston TX2|
	|----|
	|`docker run -it bitlinc/legodnn:aarch64-1.0`|

**Note! You should specify a cbc path for some devices in the `init` method of `online/scaling_optimizer.py`,like this:**
```python
pulp_solver=pulp.COIN_CMD(path="/usr/bin/cbc",msg=False, gapAbs=0)
```
	
**if your device does not  have a cbc command in `/usr/bin`,you should run `apt-get install  coinor-cbc` to install it.**
   
## 3 Repository of DNNs in vision tasks
### 3.1 Supported models 


**Image classfication**
  
  ||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|[Demo](example/image_classification/legodnn_resnet_test.py)|
  |&#9745;|[MobileNetV2 (CVPR'2018)](https://arxiv.org/abs/1801.04381)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)||
  |&#9745;|[ResNeXt (CVPR'2017)](https://arxiv.org/abs/1611.05431)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|
  |&#9745;|[InceptionV3(CVPR'2016)](https://ieeexplore.ieee.org/document/7780677/)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|[Demo](example/image_classification/legodnn_inceptionv3_test.py)|
  |&#9745;|[WideResNet (BMVC'2016)](https://dx.doi.org/10.5244/C.30.87)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|
  |&#9745;|[RAN (CVPR'2017)](https://doi.org/10.1109/CVPR.2017.683)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|
  |&#9745;|[CBAM (ECCV'2018)](https://doi.org/10.1007/978-3-030-01234-2_1)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|[Demo](example/image_classification/legodnn_cbam_r18_test.py)|
  |&#9745;|[SENet (CVPR'2018)](https://ieeexplore.ieee.org/document/341010)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|
  |&#9745;|[VGG (ICLR'2015)](http://arxiv.org/abs/1409.1556)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)|

 **Obejct detection**
 ||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[Fast R-CNN (NIPS'2015)](https://ieeexplore.ieee.org/abstract/document/7485869)|[PARSCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC)|[Demo](example/object_detection/legodnn_faster_rcnn_test.py)|
  |&#9745;|[YOLOv3 (CVPR'2018)](https://arxiv.org/abs/1804.02767)|[PARSCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC)|[Demo](example/object_detection/legodnn_yolov3_test.py)|
  |&#9745;|[FreeAnchor (NeurIPS'2019)](https://arxiv.org/abs/1909.02466)|[PARSCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC)|[Demo](example/object_detection/legodnn_freeanchor_test.py)|


**Semantic segmentation**
||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[FCN (CVPR'2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)|[PARSCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC)|[Demo](example/semantic_segmentation/legodnn_fcn_r18_test.py)|
  |&#9745;|[U-Net (MICCAI'2016)](https://arxiv.org/abs/1505.04597)|[DRIVE](https://drive.grand-challenge.org/)|[Demo](example/semantic_segmentation/legodnn_fcn_unet_s5_test.py)|
  |&#9745;|[DeepLab v3 (ArXiv'2017)](https://arxiv.org/abs/1706.05587)|[PARSCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC)|[Demo](example/semantic_segmentation/legodnn_deeplabv3_r18_test.py)|

**Anomaly detection**
||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[GANomaly (ACCV'2018)](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)|[Coil100](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)|[Demo](example/anomaly_detection/legodnn_ganomaly_test.py)|
  |&#9745;|[GPND (NIPS'2018)](https://arxiv.org/abs/1807.02588)|[CLatech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)|[Demo](example/anomaly_detection/legodnn_gpnd_test.py)|
  |&#9745;|[Self-Training (CVPR'2020)](Self-trainedDeepOrdinalRegressionforEnd-to-EndVideoAnomalyDetection)|[UCSD-Ped1](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)|[Demo](example/anomaly_detection/legodnn_ornet_test.py)

**Pose estimation**
||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[DeepPose (CVPR'2014)](https://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html)|[MPII](http://human-pose.mpi-inf.mpg.de/#overview)|[Demo](example/pose_estimation/legodnn_deeppose_r50_test.py)|
  |&#9745;|[SimpleBaselines2D (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)|[MPII](http://human-pose.mpi-inf.mpg.de/#overview)|[Demo](example/pose_estimation/legodnn_simplebaseline_r50_test.py)|


 **Action recognition**
 ||Model Name|Data|Script|
  |--|--|--|--|
  |&#9745;|[TSN (ECCV'2016)](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)|[HDMB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)|[Demo](example/action_recognition/legodnn_tsn_r50_test.py)|
  |&#9745;|[TRN (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)|[HDMB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)|[Demo](example/action_recognition/legodnn_trn_r50_test.py)|

 
  
  




### 3.3 How to implement new models in LegoDNN

The model have particular training need to implement a custom model manager based on  AbstractModelManager in package `legodnn.common.manager.model_manager.abstract_model_manager`.

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

## 4 Demo video and experiment data


### 4.1 Demo Video





https://user-images.githubusercontent.com/73862727/149520527-50c26e84-cd30-426e-94ca-a0886d104386.mp4

### 4.2 Experiment data

#### 4.2.1 Experiment setting

|Device|Models and data|Baseline|
|--|--|--|
|Ubuntu 18.04.4 LTS<br>Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz<br>Quadro RTX8000|ResNet18 on CIFAR100<br>MobileNetV2 on CIFAR100<br>GPNomaly on Coil100<br>ResNet18 on UCSD Ped1<br>Faster-RCNN-ResNet50 on PARSCAL VOC2007<br>YoloV3-DarkNet53 on PARSCAL VOC2007<br>FCN-ResNet18 on PARSCAL VOC2012<br>DeepPose-ResNet18 on MPII<br>TSN-ResNet18 on HDMB51<br>TRN-ResNet18 on HDMB|Filter Pruning<br>Low Rank Decomposition<br>Knowledge Distillation<br>NestDNN<br>US-Net<br>FN3-channel<br>OFA

#### 4.2.2 Experiment result

![实验图](https://user-images.githubusercontent.com/73862727/149600656-f9c1f1a4-a503-4b1e-94c3-8454b6aeeb92.png)


## 5 Project member and contact information

### 5.1 Project member

#### 5.1.1 Beijing Institute of Technology
Rui Han, Qinglong Zhang, Gaofeng Xin, Xinyu Guo, Yuxiao Liu, Chi Harold Liu, Guoren Wang

#### 5.1.2 TU Delft
Lydia Y.~Chen

#### 5.1.3 Midea Group
Jian Tang

### 5.2 Contact information
Rui Han: 379068433@qq.com

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

**1.0.0** was released in 2021.12.20：

  Implement basic functions
  

  
