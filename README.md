# cv-dl-llm-summer-study

Summer study notes and experiments on **Computer Vision, Deep Learning and LLM**.

本项目用于记录暑期学习期间的计算机视觉、深度学习与大语言模型学习过程，包括理论学习、代码实践、模型训练和实验分析。

---

# 一、学习目标

## 1. Computer Vision

学习计算机视觉基础知识与常用方法：

- OpenCV图像处理
    
- 图像预处理
    
- 传统目标检测基础
    
- CNN图像分类
    
- 经典CNN网络
    
- 迁移学习
    
- 模型评估与实验分析
    

## 2. Deep Learning

学习基于 PyTorch 的深度学习基本流程：

- Tensor
    
- Dataset / DataLoader
    
- MLP
    
- CNN
    
- Loss / Optimizer
    
- Backpropagation
    
- Learning Rate / Batch Size
    
- 数据增强
    
- Checkpoint
    
- 模型评估
    
- 实验复现
    

## 3. LLM

学习大语言模型基础知识与实际调用：

- Transformer
    
- Attention
    
- Token / Embedding
    
- Prompt
    
- Context Window
    
- Temperature
    
- Hallucination
    
- LLM API
    
- 本地LLM推理
    

## 4. 综合实践

完成一个 Fashion-MNIST 图像分类项目，并逐步进行模型对比与实验分析：

**Fashion-MNIST + MLP + CNN**

项目流程：

```text
数据集
  ↓
数据预处理
  ↓
模型训练
  ↓
验证
  ↓
模型保存
  ↓
测试
  ↓
实验对比
  ↓
结果分析
```

---

# 二、实验环境

## 开发环境

- Windows
    
- Python 3.11
    
- VS Code
    
- Git / GitHub
    

## 主要Python库

- PyTorch
    
- torchvision
    
- NumPy
    
- OpenCV
    
- Matplotlib
    
- scikit-learn
    
- Pillow
    

## LLM相关

- DeepSeek API
    
- Ollama
    
- DeepSeek-R1 7B
    

---

# 三、项目结构

```text
cv-dl-llm-summer-study/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── notes/
│   ├── week01_opencv.md
│   ├── week02_image_processing.md
│   ├── week03_deep_learning.md
│   ├── week04_mlp.md
│   ├── week05_cnn.md
│   ├── week06_classic_cnn.md
│   ├── week07_engineering.md
│   ├── week08_llm.md
│   ├── week09_project.md
│   ├── week10_baseline.md
│   └── week11_experiments.md
│
├── image_processing/
│
└── projects/
    └── fashion_mnist/
        ├── dataset.py
        ├── model.py
        ├── train.py
        ├── test.py
        ├── configs/
        └── outputs/
```

---

# 四、环境配置

创建Python虚拟环境：

```bash
python -m venv .venv
```

激活环境：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch：

```bash
python -c "import torch; print(torch.__version__)"
```

检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 五、运行方式

## Fashion-MNIST CNN Baseline

进入项目目录：

```bash
cd projects/fashion_mnist
```

训练模型：

```bash
python train.py
```

测试模型：

```bash
python test.py
```

项目主要涉及：

```text
训练
 ↓
验证
 ↓
保存最佳模型
 ↓
测试
 ↓
Accuracy / Loss
 ↓
结果分析
```

---

# 六、学习进度

## Week01 OpenCV基础

### 完成内容

- OpenCV环境配置
    
- 图片读取与保存
    
- 图像尺寸与通道分析
    
- BGR / Gray / HSV颜色空间
    
- 图像缩放
    

### 主要成果

建立 Python + OpenCV 实验环境，理解图片在计算机中的数组表示，完成基础图像处理流程。

---

## Week02 图像预处理与目标检测基础

### 完成内容

- 灰度化
    
- 高斯滤波
    
- 图像二值化
    
- 阈值分割
    
- 腐蚀、膨胀、开运算、闭运算
    
- 轮廓提取
    
- 轮廓面积与中心点计算
    
- 基于轮廓的目标计数
    

### 主要成果

理解：

```text
原始图像
 ↓
灰度图
 ↓
二值图
 ↓
目标轮廓
 ↓
目标计数
```

掌握传统计算机视觉中的基础图像处理方法。

---

## Week03 深度学习训练流程

### 完成内容

- PyTorch环境配置
    
- Tensor基础
    
- 数据集与数据划分
    
- 线性回归
    
- MSE Loss
    
- 梯度下降
    
- 反向传播
    
- SGD优化器
    
- Learning Rate
    
- Batch Size
    
- Loss曲线
    

### 主要成果

理解深度学习基本训练流程：

```text
数据输入
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Update
```

完成 `y = 3x + 2` 线性回归实验，并分析学习率和 Batch Size 对训练过程的影响。

---

## Week04 MLP与图像分类基础

### 完成内容

- Linear
    
- ReLU
    
- Dropout
    
- MNIST数据集
    
- Train / Validation / Test
    
- CrossEntropyLoss
    
- MLP图像分类
    
- 不同隐藏层规模对比
    
- Learning Rate / Batch Size实验
    

### 主要成果

完成基于 PyTorch 的 MNIST MLP 分类实验，理解全连接神经网络进行图像分类的基本流程。

---

## Week05 CNN与图像分类

### 完成内容

- CNN
    
- Conv2D
    
- Kernel
    
- Feature Map
    
- Padding / Stride
    
- Pooling
    
- CNN图像分类
    
- Accuracy与Loss
    
- 预测结果分析
    

### 主要成果

完成 MNIST CNN 分类模型，理解：

- 卷积提取图像局部特征
    
- Pooling降低特征维度
    
- CNN保留图像空间结构
    

并认识到 CNN 通常比直接展平图像的 MLP 更适合图像任务。

---

## Week06 经典CNN架构与迁移学习

### 完成内容

学习经典卷积神经网络：

- AlexNet
    
- VGG
    
- GoogLeNet
    
- ResNet
    

学习：

- BatchNorm
    
- Residual Connection
    

完成 ResNet18 迁移学习实验：

- ImageNet预训练模型加载
    
- 冻结特征提取层
    
- 替换分类头
    
- 灰度图转换三通道
    
- ImageNet Normalize
    
- 模型训练与测试
    

### 主要成果

理解经典 CNN 的发展过程以及迁移学习基本流程：

```text
Pretrained Model
 ↓
Freeze Feature Layers
 ↓
Replace Classifier
 ↓
Train New Task
```

---

## Week07 PyTorch工程化训练与实验复现

### 完成内容

- Dataset / DataLoader
    
- 模块化训练
    
- Model / Train / Test分离
    
- 最佳模型保存
    
- Training Log
    
- `history.csv`
    
- Accuracy曲线
    
- Confusion Matrix
    
- Random Seed
    
- CUDA随机种子
    
- cuDNN Deterministic
    

### 主要成果

建立工程化训练流程：

```text
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
Best Model
 ↓
Test
 ↓
Analysis
```

同时学习：

- 数据增强
    
- 配置文件
    
- Checkpoint
    
- TensorBoard
    

其中数据增强、配置文件和 TensorBoard 主要作为工程化扩展内容学习，后续在 Fashion-MNIST 项目中继续实践。

---

## Week08 大语言模型基础

### 完成内容

学习：

- LLM与传统深度学习模型
    
- Transformer
    
- Attention
    
- Query / Key / Value
    
- Token / Tokenizer
    
- Embedding
    
- Prompt
    
- Context Window
    
- Temperature
    
- Hallucination
    

### 主要成果

理解 LLM 基本工作流程：

```text
文本
 ↓
Tokenization
 ↓
Embedding
 ↓
Transformer / Attention
 ↓
预测下一个Token
 ↓
生成文本
```

同时认识到：

**语言流畅不代表事实正确。**

---

## Week09 视觉项目确定与LLM API实践

### 完成内容

确定最终视觉项目：

**基于 CNN 的服装图像分类**

使用 Fashion-MNIST：

- 28×28灰度图
    
- 10个类别
    
- Train / Validation / Test划分
    
- ToTensor
    
- Normalize
    
- DataLoader
    

实际实验规模后续根据训练速度进行了调整。

完成 DeepSeek API 基础实践：

- `.env`
    
- API Key
    
- OpenAI风格Client
    
- System Prompt / User Prompt
    
- Prompt Template
    
- JSON结构化输出
    
- Python JSON解析
    

### 主要成果

确定：

**Fashion-MNIST + MLP基线 + CNN分类模型**

并完成项目数据加载部分，为后续模型训练做准备。

---

## Week10 视觉项目Baseline与LLM实践

### 完成内容

完成 Fashion-MNIST CNN Baseline：

- Dataset / DataLoader
    
- CNN模型
    
- CrossEntropyLoss
    
- Adam
    
- Train / Validation / Test
    
- 最佳模型保存
    
- Accuracy / Loss曲线
    
- Test Accuracy
    
- Confusion Matrix
    

当前实验使用较小的数据规模进行训练：

```text
Train：5000
Validation：1000
Test：10000
```

CNN结构：

```text
Input
 ↓
Conv2d(1 → 32)
 ↓
ReLU + MaxPool
 ↓
Conv2d(32 → 64)
 ↓
ReLU + MaxPool
 ↓
Flatten
 ↓
Linear(3136 → 128)
 ↓
ReLU
 ↓
Linear(128 → 10)
```

同时完成：

- DeepSeek API实践
    
- Ollama配置
    
- DeepSeek-R1 7B本地推理
    

### 主要成果

完成 Fashion-MNIST CNN Baseline，建立：

```text
Training
 ↓
Validation
 ↓
Best Model
 ↓
Test
 ↓
Accuracy / Loss / Confusion Matrix
```

的完整训练流程。

---

## Week11 模型对比与实验分析

### 学习内容

开始学习视觉模型实验中的常见分析方法：

- 消融实验
    
- 参数对比
    
- 数据增强对比
    
- 类别级准确率
    
- 混淆矩阵
    
- 错误样例分析
    

理解不同实验方法的基本目的：

- **消融实验**：分析某个模型组件是否有效
    
- **参数对比**：分析不同训练参数的影响
    
- **数据增强对比**：分析数据增强对模型泛化能力的影响
    
- **类别级准确率**：分析模型对不同类别的识别能力
    
- **混淆矩阵**：分析类别之间的错误预测关系
    
- **错误样例分析**：观察模型具体在哪些图片上出错
    

### 实验准备

在 CNN Baseline 的基础上，开始准备：

- MLP vs CNN
    
- 无数据增强 vs 有数据增强
    
- 不同学习率等参数对比
    
- CNN 与 ResNet18 对比
    

同时开始完善错误样例保存与可视化代码，为后续 Error Analysis 做准备。

### 主要成果

建立从整体指标到具体错误的分析思路：

```text
Test Accuracy
      ↓
类别级准确率
      ↓
混淆矩阵
      ↓
错误样例
      ↓
分析模型错误
```

目前部分对比实验仍在进行中，尚未将所有实验结果作为最终结论。

---

# 七、阶段成果

## Computer Vision

完成从传统图像处理到深度学习视觉任务的学习：

```text
OpenCV
 ↓
图像预处理
 ↓
传统目标检测基础
 ↓
MLP
 ↓
CNN
 ↓
经典CNN架构
 ↓
迁移学习
 ↓
Fashion-MNIST项目
```

## Deep Learning

掌握和实践：

- PyTorch基础
    
- Tensor
    
- Dataset / DataLoader
    
- MLP
    
- CNN
    
- Loss
    
- Optimizer
    
- Backpropagation
    
- Learning Rate
    
- Batch Size
    
- 模型保存
    
- 模型评估
    
- 实验复现
    

同时学习数据增强、Checkpoint、配置文件、TensorBoard等工程化内容。

## LLM

完成从理论学习到实际调用：

```text
Transformer
 ↓
Attention
 ↓
Token / Embedding
 ↓
Prompt
 ↓
API调用
 ↓
本地LLM推理
```

完成 DeepSeek API 和 Ollama 本地推理实践。

## 综合项目

目前已完成 Fashion-MNIST CNN Baseline，并开始进行后续模型对比和结果分析。

当前项目进展：

```text
Fashion-MNIST
      ↓
数据预处理
      ↓
CNN Baseline       
      ↓
MLP对比            
      ↓
数据增强对比        
      ↓
参数对比            
      ↓
迁移学习对比        
      ↓
类别级准确率        
      ↓
错误样例分析        
```

项目目前处于：

**Baseline完成 → 对比实验与结果分析阶段。**