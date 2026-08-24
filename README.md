# cv-dl-llm-summer-study
Summer study notes and experiments on Computer Vision, Deep Learning and LLM.
# 学习进度

## Week01 OpenCV基础

完成内容：

- OpenCV环境配置
- 图片读取与保存
- 图像尺寸与通道分析
- BGR、Gray、HSV颜色空间转换
- 图像缩放实验

主要成果：

- 建立Python+OpenCV实验环境
- 理解图片在计算机中的数组表示
- 完成基础图像处理流程

详细记录：
- notes/week01_opencv.md
## Week02 图像预处理与目标检测基础

### 完成内容：

- 图像灰度化处理
- 高斯滤波与图像去噪
- 图像二值化处理
- 阈值分割与反向二值化
- 形态学操作：
    - 腐蚀（Erosion）
    - 膨胀（Dilation）
    - 开运算（Opening）
    - 闭运算（Closing）
- 图像轮廓提取（Contour Extraction）
- 轮廓面积计算与筛选
- 轮廓中心点计算
- 基于轮廓的目标计数实验
### 主要成果：

- 理解图像从**原始像素 → 二值图 → 目标轮廓**的处理流程
- 掌握传统计算机视觉中的基础目标检测方法：
## Week03 深度学习训练流程

完成内容：
- PyTorch环境配置
- 张量（Tensor）基础操作
- 数据集与训练数据划分理解
- 线性回归模型搭建
- Loss函数（MSE）计算
- 梯度下降优化方法
- 反向传播（Backpropagation）流程理解
- 使用SGD优化器进行模型训练
- 学习率（Learning Rate）调整实验
- Batch Size调整实验
- Loss曲线绘制与训练效果分析

主要成果：
- 理解深度学习模型的基本训练流程：
  数据输入  
  ↓  
  前向传播（Forward）  
  ↓  
  计算损失（Loss）  
  ↓  
  反向传播（Backward）  
  ↓  
  参数更新（Update）
- 
- 掌握神经网络训练中的核心概念：
  - 模型参数（Weight、Bias）
  - 损失函数（Loss Function）
  - 梯度（Gradient）
  - 学习率（Learning Rate）
  - Batch训练方式

- 完成基于PyTorch的线性回归实验：
  - 使用 Linear 模型拟合 y=3x+2 数据
  - 通过梯度下降自动优化模型参数
  - 观察训练过程中 Loss 的下降趋势

- 分析不同训练参数对模型效果的影响：
  - 学习率过小时：
    - 参数更新速度慢
    - Loss下降缓慢

  - 学习率合适时：
    - Loss稳定下降
    - 模型快速收敛

  - Batch Size变化：
    - 影响梯度计算稳定性
    - 影响训练速度和Loss变化曲线
# Week04 多层感知机（MLP）与图像分类基础

## 完成内容

### MLP模型结构学习

- 全连接层（Linear）
- 激活函数（ReLU）
- Dropout正则化

### MNIST手写数字分类实验

完成基于PyTorch的MLP分类模型：

- MNIST数据集加载
- Tensor转换
- 训练集 / 验证集 / 测试集划分

### 模型训练流程

- 前向传播（Forward）
- Loss计算（Cross Entropy）
- 反向传播（Backward）
- 参数更新（Update）

### 模型实验

- 比较不同隐藏层规模对模型效果的影响
- 调整学习率（Learning Rate）
- 调整Batch Size
- 使用Dropout降低过拟合

## 主要成果

### 理解MLP图像分类流程

- 图片输入与数据预处理
- 神经网络特征提取
- 分类结果预测

### 掌握深度学习核心概念

- 多层感知机（MLP）
- ReLU激活函数
- Cross Entropy Loss
- Dropout
- 训练集 / 验证集 / 测试集划分

### 完成实验

基于PyTorch完成MNIST分类任务：

- 搭建MLP模型
- 完成模型训练与测试
- 绘制Loss变化曲线
- 分析不同参数对模型性能的影响
# Week05 卷积神经网络（CNN）与图像分类

## 完成内容

### CNN基础学习

- 卷积神经网络（CNN）
- 卷积层（Conv2D）
- 卷积核（Kernel）
- 特征图（Feature Map）
- Padding、Stride、Pooling

### CNN模型训练

完成基于PyTorch的CNN图像分类实验：

- MNIST数据集加载与预处理
- CNN模型搭建
- 卷积层、池化层、全连接层组合
- 使用Cross Entropy Loss进行分类训练
- 使用Adam优化器更新参数

### 模型测试与分析

完成：

- 测试集准确率计算
- Loss曲线绘制
- 正确预测样例可视化
- 错误预测样例分析


## 主要成果

### 理解CNN图像分类流程

掌握：

- 卷积提取图像局部特征
- 池化降低特征维度
- 全连接层完成分类任务

### 理解CNN相比MLP的优势

- 保留图像空间结构信息
- 自动学习图像特征
- 更适合图像分类任务

### 完成实验

基于PyTorch完成MNIST CNN分类：

- 搭建基础CNN网络
- 完成模型训练与测试
- 输出测试准确率
- 保存训练曲线与预测结果
# Week06 经典CNN架构与迁移学习

## 完成内容

### 经典网络结构学习
学习经典卷积神经网络：
- AlexNet
- VGG
- GoogLeNet（Inception）
- ResNet
理解不同网络结构在深度学习发展中的作用：
- 提升网络深度
- 优化特征提取能力
- 降低参数量
- 解决深层网络训练困难问题
### BatchNorm与残差连接学习
学习：
- Batch Normalization（BatchNorm）
- 残差连接（Residual Connection）
理解：
- BatchNorm通过规范化特征分布提高训练稳定性
- 残差连接通过Shortcut改善梯度传播
- ResNet解决深层网络训练中的梯度消失问题
### 迁移学习实验
完成基于PyTorch的ResNet18迁移学习实验：
- 加载ImageNet预训练模型
- 冻结原始网络参数
- 修改分类层适配新任务
- 使用MNIST数据集进行分类训练
### 数据处理与模型训练
完成：
- 图片Resize调整
- 灰度图转换三通道
- ImageNet标准化
- ResNet18模型训练
- 测试准确率计算
- Accuracy曲线绘制
## 主要成果
### 理解经典CNN发展过程
掌握：
- AlexNet推动深度学习视觉任务发展
- VGG通过堆叠卷积构建深层网络
- GoogLeNet通过Inception模块提升特征提取效率
- ResNet通过残差连接优化深层网络训练
### 理解迁移学习流程
掌握：
- 预训练模型加载
- 特征提取层冻结
- 分类头替换
- 新任务微调训练
### 完成实验
基于PyTorch完成ResNet18迁移学习：
- 使用ImageNet预训练权重
- 完成MNIST分类任务
- 输出测试准确率
- 保存模型参数与训练记录
- 
# Week07 PyTorch工程化训练与实验复现
## 理论学习
- Dataset / DataLoader：理解数据集管理、批量加载、Batch Size 与 Shuffle
- 数据增强：理解通过随机变换增加数据多样性，提高模型泛化能力
- 配置文件：理解将实验参数与代码逻辑分离，便于管理不同实验
- Checkpoint：理解训练过程中的模型存档，以及最佳模型与断点续训的区别
- 日志：理解记录训练过程、实验参数和运行状态
- TensorBoard：理解使用可视化面板观察 Loss、Accuracy、Learning Rate 等训练指标
- 混淆矩阵：理解通过真实类别与预测类别的对应关系分析模型分类结果
- 实验复现：理解随机种子、确定性设置与实验环境对结果复现的影响
## 完成内容
### Dataset / DataLoader
- 使用 `Dataset` 管理数据集
- 使用 `DataLoader` 进行Batch训练
- 完成训练集 / 验证集 / 测试集划分
### 模块化训练
将训练代码拆分为：
- `dataset.py`：数据集与DataLoader
- `model.py`：模型定义
- `train.py`：训练与验证
- `test.py`：测试与结果分析
### 模型保存与日志
完成：
- 保存验证集表现最佳的模型
- 记录训练过程日志
- 保存训练历史 `history.csv`
- 绘制训练 / 验证 Accuracy 曲线

### 实验结果分析
完成：
- 测试集 Accuracy 计算
- 10×10 混淆矩阵生成
- 混淆矩阵可视化
- 分析模型具体分类错误
### 实验复现
学习并实践：
- Python / NumPy / PyTorch 随机种子
- CUDA 随机种子
- cuDNN 确定性设置
通过固定实验参数和随机种子，提高实验结果的可重复性。
## 主要成果

### 掌握工程化训练流程
```
Dataset
   ↓
DataLoader
   ↓
Model
   ↓
Train
   ↓
Validation
   ↓
保存最佳模型
   ↓
Test
   ↓
结果分析
```
### 完成实验

基于 PyTorch 完成 ResNet18 迁移学习实验：
- 使用 ImageNet 预训练 ResNet18
- 完成 MNIST 分类训练
- 保存 `best_model.pth`
- 输出 Test Accuracy
- 生成混淆矩阵
- 保存训练日志和实验记录
### 学习内容
了解并学习：
- 数据增强
- 配置文件
- Checkpoint
- TensorBoard
其中上述内容作为工程化深度学习的扩展知识进行学习，**本周实验未实际实现数据增强、配置文件和 TensorBoard**。

# Week08 大语言模型基础

## 理论学习

### 大模型与普通深度学习模型

- 理解 LLM 与传统深度学习模型的区别
- 理解 Transformer 与大模型的关系
- 理解分类模型与生成式模型的任务差异
- 了解 LLM 通过 Prompt 完成多种任务
### Transformer 与 Attention

- Transformer 基本结构
- Attention 注意力机制
- Query / Key / Value
- 理解 Attention 对不同 Token 进行动态关注

### Token 与 Embedding

- Token 与 Tokenizer
- Token ID
- Embedding 向量表示
- 理解 Token 到向量表示的转换过程

### Prompt 与上下文窗口

- Prompt 基本概念
- Prompt 对模型输出的影响
- 上下文与上下文窗口
- 理解上下文窗口对模型信息处理范围的限制

### Temperature 与幻觉

- Temperature 温度参数
- 生成随机性与多样性
- 幻觉（Hallucination）
- 理解“语言流畅不代表事实正确”的模型能力边界

## 主要成果

### 理解 LLM 基本工作流程

建立从文本输入到文本生成的基本认识：

```text
输入文本
   ↓
Tokenization
   ↓
Token
   ↓
Embedding
   ↓
Transformer / Attention
   ↓
预测下一个 Token
   ↓
不断生成 Token
   ↓
完整输出
```

### 理解传统模型与生成式大模型的区别

掌握：

- 分类模型：输入 → 预定义类别
- 生成模型：输入 → 根据上下文生成内容
- 传统模型通常针对特定任务进行训练
- LLM 可以通过不同 Prompt 完成问答、总结、翻译、改写等任务

### 理解 LLM 的主要能力

了解 LLM 可以完成：

- 文本生成
- 文本理解
- 信息提取与分类
- 摘要与改写
- 翻译
- 问答与一定程度的推理
- 多轮对话
- 根据 Prompt 完成不同任务

### 理解 LLM 的能力边界

认识到：

- LLM 不能保证生成内容一定正确
- 可能产生事实错误和幻觉
- 不能保证信息始终实时准确
- 复杂推理仍可能出现错误
- Prompt 不清晰或上下文不足时可能产生错误理解

### 核心认识

理解：
**传统深度学习模型通常针对特定任务输出固定形式的结果，而 LLM 可以根据 Prompt 和上下文生成不同形式的内容。**
同时认识到：
**LLM 擅长生成合理的语言，但“合理”不等于“真实”，“流畅”也不等于“正确”。**
# Week09 视觉项目确定与 LLM API 实践

## 完成内容

### 视觉项目确定

确定最终视觉项目：

**基于 CNN 的服装图像分类**

使用 Fashion-MNIST 数据集完成服装图像分类，并计划通过 MLP 基线模型与 CNN 模型进行对比。

### 数据集与任务

完成 Fashion-MNIST 数据集调研：

- 了解数据集基本结构
    
- 确定 10 个服装与鞋类分类类别
    
- 确定图片大小为 28 × 28
    
- 理解灰度图与单通道数据形式
    
- 确定图像分类任务
    

### 数据集划分

确定：

- Train：50,000
    
- Validation：10,000
    
- Test：10,000
    

理解训练集、验证集和测试集在模型训练与最终评价中的不同作用。

### 数据预处理

完成 Fashion-MNIST 初步预处理方案：

- `ToTensor`
    
- 像素值 0～255 转换为 0～1
    
- `Normalize`
    
- 理解输入 CNN 的数据形式 `[1, 28, 28]`
    

### 数据加载

完成 `data_loader.py` 初稿：

- Fashion-MNIST 数据集加载
    
- Train / Validation / Test 划分
    
- DataLoader 构建
    
- Batch Size 设置
    
- 数据 Shape 与数据范围检查
    

### LLM API 实践

使用 **DeepSeek API** 完成大模型 API 基础调用实验：

- `.env` 环境变量管理
    
- API Key 读取
    
- OpenAI 风格 API Client 创建
    
- 最小聊天示例
    
- System Prompt / User Prompt
    
- Prompt Template
    
- 结构化 JSON 输出
    
- Python `json.loads()` 解析模型结果
    

## 主要成果

### 确定最终视觉项目

确定后续项目使用：

**Fashion-MNIST + MLP 基线 + CNN 分类模型**

计划通过：

- Accuracy
    
- Confusion Matrix
    

比较不同模型的分类效果。

### 掌握数据处理流程

建立 Fashion-MNIST 的基本处理流程：

```text
Fashion-MNIST
    ↓
Train / Validation / Test
    ↓
ToTensor
    ↓
Normalize
    ↓
DataLoader
    ↓
输入模型
```

完成 `data_loader.py` 初稿，为后续模型训练做准备。

### 初步掌握 LLM API 调用

理解基本调用流程：

```text
Python
   ↓
Prompt
   ↓
DeepSeek API
   ↓
模型生成
   ↓
文本 / JSON
   ↓
Python 程序处理
```

同时理解 Prompt Template 可以提高 Prompt 的复用性，结构化输出可以让 LLM 生成的内容进一步被程序处理。

### 本周学习重点

本周开始从前期的**基础知识学习**进入**实际项目准备阶段**：

- 确定最终视觉项目方向
    
- 完成数据集与预处理设计
    
- 开始搭建项目数据加载部分
    
- 同时完成 LLM API 的基础调用实践
    

后续主要继续完成 MLP 基线、CNN 模型训练以及模型结果分析。
# Week10 Fashion-MNIST视觉项目Baseline实现

## 完成内容

### CNN Baseline搭建

完成基于 PyTorch 的 Fashion-MNIST CNN 分类 Baseline：

- 搭建基础 CNN 模型
- 使用两层卷积层提取图像特征
- 使用 MaxPool 降低特征图尺寸
- 使用全连接层完成 10 分类
- 使用 `CrossEntropyLoss` 计算分类损失
- 使用 Adam 优化器进行训练

### 数据处理

完成 `dataset.py`：

- Fashion-MNIST 数据集加载
- Train / Validation / Test 数据划分
- `ToTensor` 数据转换
- `Normalize` 数据标准化
- DataLoader 构建
- Batch Size 设置
- 数据 Shape、Label 和数据范围检查

当前实验数据规模：

- Train：5000
- Validation：1000
- Test：10000

### 模型结构

完成 `model.py`：

- `Conv2d(1 → 32)`
- `ReLU`
- `MaxPool`
- `Conv2d(32 → 64)`
- `ReLU`
- `MaxPool`
- `Flatten`
- `Linear(3136 → 128)`
- `ReLU`
- `Linear(128 → 10)`

### 工程化训练

完成 `train.py`：

- 自动选择 CPU / CUDA
- 固定随机种子
- Train / Validation 训练流程
- Loss 与 Accuracy 统计
- 根据 Validation Accuracy 保存最佳模型
- 保存 `best_model.pth`
- 保存 `history.csv`
- 绘制 Loss 曲线
- 绘制 Accuracy 曲线

### 模型测试与分析

完成 `test.py`：

- 加载训练得到的最佳模型
- 在测试集上进行推理
- 计算 Test Accuracy
- 统计正确预测数量
- 生成 10×10 混淆矩阵
- 保存 `confusion_matrix.png`

## 主要成果

### 建立完整视觉分类流程

```
Fashion-MNIST
      ↓
Dataset
      ↓
DataLoader
      ↓
CNN
      ↓
Training
      ↓
Validation
      ↓
保存最佳模型
      ↓
Test
      ↓
Accuracy / Loss / Confusion Matrix
```

### 初步完成视觉项目Baseline

完成从数据读取、模型构建、训练、验证到测试和结果可视化的完整流程。

项目目前已经从前期的**数据集调研和代码初稿**进入实际的**模型训练与实验阶段**。

## 后续计划

- 完成 MLP Baseline
- 对比 MLP 与 CNN 的分类效果
- 分析不同模型的 Accuracy 与 Loss
- 分析 Fashion-MNIST 混淆矩阵
- 根据实验结果总结 CNN 在图像分类任务中的优势