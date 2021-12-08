<div align="center">


<img src="https://user-images.githubusercontent.com/20336673/145025177-6dd4d49b-65ed-457d-84a0-d9e716d85039.png" width="375"/>

![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
![docs](https://img.shields.io/badge/docs-latest-blue)
![license](https://img.shields.io/badge/license-Apache2.0-green)

![图片3](https://user-images.githubusercontent.com/20336673/145015327-3fbfe409-2fc8-48e8-ac4f-d7f36b256dc9.png)


</div>
 

## 简介

  LegoDNN（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)）是一个针对模型缩放问题的轻量级、块粒度、可伸缩的解决方案。本项目是一个对LegoDNN的基于PyTorch的实现。
  
  **主要特性**
- **模块化设计**

  本项目将LegoDNN的抽块、再训练等过程解耦成各个模块，通过组合不同的模块组件，用户可以更便捷的对自己的自定义模型Lego化。
  
- **块的自动化抽取**
    
    本项目实现了通用的块的抽取算法（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)），对于图像分类、目标检测、语义分割、姿态估计、行为识别、异常检测等类型的模型均可以通过算法，自动找出其中的块用于再训练。

## 项目整体架构
<div align="center">
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

```
pip install legodnn
```


## docker

使用镜像

|树莓派4B|Jeston|
|----|----|
|`docker run -it lincbit/legodnn:raspberry4B-1.0`|`docker run -it lincbit/legodnn:jetsontx2-1.0`|

## 使用




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
  
