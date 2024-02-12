# Assignment 2

## Exercise 1. 全连接神经网络 Fully-connected Neural Network

    In this exercise we will implement fully-connected networks using a more modular approach. For each layer we will implement a`forward` and a `backward` function.

    In addition to implementing fully-connected networks of arbitrary depth, we will also explore different update rules for optimization, and introduce Dropout as a regularizer and Batch/Layer Normalization as a tool to more efficiently optimize deep networks.

    正常按照提示实现即可。

## Exercise 2. 批归一化 Batch Normaliztion

使深度网络更容易训练的一种方法是使用更复杂的优化过程，如SGD+动量、RMSProp或Adam。另一种策略是改变网络的架构，使其更容易训练。 其中一种思路是批归一化，这是由[1]在2015年提出的。

这个想法相对简单。 **机器学习方法在其输入数据由零均值和单位方差的不相关特征组成时往往效果更好。** 当训练神经网络时，我们可以在将数据馈送到网络之前对其进行预处理，以显式地去相关其特征；这将确保网络的第一层看到遵循良好分布的数据。然而，即使我们对输入数据进行了预处理，网络较深层的激活值可能不再是不相关的，也不再具有零均值或单位方差，因为它们是从网络的较早层输出的。更糟糕的是，在训练过程中，每一层网络的特征分布都会随着每一层的权重更新而发生变化。

[1]的作者假设，深度神经网络内部特征的分布变化可能会使训练深度网络变得更加困难。为了解决这个问题，[1]提出在网络中插入批归一化层。**在训练时，批归一化层使用一个小批量的数据来估计每个特征的均值和标准差。然后，使用这些估计的均值和标准差对小批量的特征进行居中和归一化。在训练过程中，保持这些均值和标准差的运行平均值，并在测试时使用这些运行平均值对特征进行居中和归一化。**

这种归一化策略可能会降低网络的表示能力，因为对于某些层来说，具有非零均值或非单位方差的特征可能是最优的。为此，批归一化层包括每个特征维度的可学习的偏移和缩放参数。

$$
\mu_j = \frac{1}{N}\sum_{i=1}^{N}x_{i,j} ~~~~~~~ 每个像素点上的均值
$$

$$
\sigma_j^2 = \frac{1}{N}\sum^N_{i=1}(x_{i,j}-\mu_j)^2  ~~~~~~~每个像素点h上的方差
$$

$$
\hat{x}_{i,j} = \frac{x_{i,j}-\mu_j}{\sqrt{\sigma^2+\epsilon}} ~~~~~~~~~~~~~~~~ 正则化的每个像素的值
$$

$$
y_{i,j} = \gamma \hat{x}_{i,j}+\beta_j  ~~~~~~~~~~~~最后输出
$$

## Exercise 3. Dropout

Dropout是一种通过在前向传播过程中随机将一些输出激活设置为零来对神经网络进行正则化的技术。

直接用随机数生成每次要丢掉的神经元。若保留的概率为p，之后前向传播的权重要除以p来保证梯度（权值和）。
