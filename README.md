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