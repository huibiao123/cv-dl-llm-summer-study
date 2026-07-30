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
