# 简单深度学习算法推导
- From-Scratch Implementation of Deep Learning Algorithms

## 项目简介
本项目旨在通过纯NumPy从零构建深度学习核心算法，完全不依赖PyTorch/TensorFlow等框架的自动微分（Automatic Differentiation）功能，手动完成前向传播（Forward Propagation）、反向传播（Backward Propagation）的数学推导与代码实现，从而深入理解神经网络的底层运行机制。

## 项目背景
作为入门学习者，跳出"调包式"学习模式，从数学原理和矩阵运算层面掌握深度学习核心逻辑：
- 摒弃高级API的自动微分（Auto-grad）依赖，全手动实现梯度传播
- 掌握神经网络核心模块的数学推导过程
- 提升高维张量（High-dimensional Tensor）运算与数值稳定性（Numerical Stability）处理能力

## 核心实现内容
### 1. 前馈神经网络 (Feedforward Neural Networks, FFN)
- 手动实现全连接层（Fully Connected Layer）的前向/反向传播
- 推导损失函数（Loss Function）对权重（Weights）和偏置（Biases）的梯度公式
- 完成二分类（Binary Classification）与多分类（Multi-class Classification）任务验证

### 2. 卷积神经网络 (Convolutional Neural Networks, CNN)
- 手动实现卷积层（Convolution Layer）与池化层（Pooling Layer）的前向/反向传播
- 理解并验证局部连接（Local Connectivity）与权重共享（Weight Sharing）核心机制
- 基于简单图像分类任务验证CNN效果

### 3. 循环神经网络 (Recurrent Neural Networks, RNN)
- 推导时间反向传播（Backpropagation Through Time, BPTT）完整数学公式
- 手动实现基础RNN单元的梯度计算与传播
- 构建序列预测模型并完成基础时序任务验证

### 4. 待扩展模块 (Upcoming)
后续将逐步补充更多深度学习核心算法的手动实现与数学推导。

## 关键挑战与解决
| 挑战 | 解决方案 |
|------|----------|
| 高维张量梯度传播 | 基于矩阵运算手动拆解梯度维度，通过数值梯度（Numerical Gradient）对比验证计算正确性 |
| 手动推导易出错 | 分步推导+小批量数据验证，确保梯度计算精准 |
| 数值稳定性问题 | 针对性处理梯度消失/爆炸、数值溢出等问题，保障模型收敛 |

## 学习收获
1. 从数学原理（Mathematical Principles）层面理解深度学习核心算法，而非仅停留在API使用层面
2. 具备阅读学术论文中复杂模型架构，并自主实现底层修改的能力
3. 熟练掌握NumPy高维张量运算与梯度计算逻辑

### 总结
1. 项目核心为**纯NumPy手动实现深度学习算法**，核心目标是理解底层数学原理，而非依赖自动微分工具；
2. 现阶段已完成FFN、CNN、RNN三大基础神经网络的完整推导与实现，后续将扩展更多算法；
3. 面向入门学习者设计，重点强化梯度推导、张量运算等核心基础能力。
