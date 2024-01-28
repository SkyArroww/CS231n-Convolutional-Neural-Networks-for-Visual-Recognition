# Assignment 1

## Exercise1. K-Nearest Neighbor Classifier

Understand the basic Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, vectorized code.

### Cifar-10 数据集

训练集图片 X_train: $50000\times32\times32\times3$

训练集标签 y_train: $50000$

测试集图片 X_test: $10000\times32\times32\times3$

测试集标签 y_test: $50000$

共有10个类型

训练时，以**5000张训练样本和500张测试样本为一组**

X_train $5000\times3072$

X_test  $ 500\times3072$

### KNN第一步：计算每个训练样本和测试样本的距离

* 用两个循环遍历测试集与训练集，暴力枚举每一对样本计算距离。 耗时15.6s
* 用一个循环遍历测试集，每个样本broadcast，然后降维求和。 耗时38s
* 不用循环，利用平方差公式拆成样本的自己的平方和与两组样本的内积。降维求和，broadcast(有一组样本求和后需变成列向量)。耗时0.1s

### KNN第二步：找出K个距离最小的训练样本，进行预测

* 利用np.argsort找出距离最小的k个样本的对应的下标，找出对应的标签
* 利用np.bincount对标签的出现次数进行计数，用np.argmax找出h出现词素最多的标签

### N-折交叉验证

* 把训练集再分成N份，每次选一份作为测试集，剩下的作为训练集。以此来找出表现最好的k

## Exercise2. 训练一个SVM(Support Vector Machine)

In this exercise you will:

- implement a fully-vectorized **loss function** for the SVM
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** using numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

### 数据处理

- 引入dev集
- 将每个图片的原始数据补上一个1当做矩阵的偏置项，这样b就不用单独训练

### 损失函数与梯度

$$
Hinge Loss:L_i = \sum_{j\neq y_i}max(0, s_j-s_{y_i} + 1)
$$

* Loss可用上式计算得出。
* 梯度即为每个像素的值

### 随机梯度下降SGD

* 每次选择Minibatch（约200个样本）进行训练和梯度下降

## Exercise3. Softmax分类器

This exercise is analogous to the SVM exercise. You will:

- implement a fully-vectorized **loss function** for the Softmax classifier
- implement the fully-vectorized expression for its **analytic gradient**

* **check your implementation** with numerical gradient

- use a validation set to **tune the learning rate and regularization** strength

* **optimize** the loss function with **SGD**
* **visualize** the final learned weights

$$
Softmax: P_i = -log(\frac{e^{s_{y_i}}}{\sum_{j}e^{s_j}})
$$

$$
Cross Entropy Loss: L_i = H_i = -log(P_i)
$$

- Softmax的数值稳定问题：在编程实现的时候容易发现直接用Softmax容易出现各种损失函数爆炸、梯度爆炸的问题，这是因为浮点数的范围是有限的。在做指数运算$e^{s_j}$的时候，如果指数过大，就直接上溢(overflow)了。为了避免上溢的问题，需要对所有的指数项$s_j$做一次均值迁移，减去其中的最大项。
- Softmax求导：

  1. $y_i = j$:

     $$
     \frac{\delta P_i}{\delta s_{j}} =\frac{\delta}{\delta s_{j}}(\frac{e^{s_{y_i}}}{\sum_{k}e^{s_k}}) = \frac{e^{s_{y_i}}\cdot \sum_{k}e^{s_k} - e^{s_{y_i}}\cdot e^{s_j}}{(\sum_{k}e^{s_k})^2} = P_i-P_i^2 = P_i(1-P_i)
     $$
  2. $y_i\neq j $

     $$
     \frac{\delta P_i}{\delta s_{j}} =\frac{\delta}{\delta s_{j}}(\frac{e^{s_{y_i}}}{\sum_{k}e^{s_k}}) = \frac{0\cdot \sum_{k}e^{s_k} - e^{s_{y_i}}\cdot e^{s_j}}{(\sum_{k}e^{s_k})^2} = -P_i \cdot P_j
     $$
- Softmax 结合 交叉熵损失函数求导：

  $$
  H(y_i,P_i) = -\sum_iy_i\cdot log(P_i), 其中y_i是one-hot向量
  $$

  $$
  \frac{\delta H}{\delta s_j} = \frac{\delta H}{\delta P_i}\frac{\delta P_i}{\delta s_j} = -\sum_iy_i\frac{1}{P_i}\cdot \frac{\delta P_i}{\delta s_j} = -\sum_{i=j}\frac{y_i}{P_i}\cdot P_i(1-P_i)-\sum_{i\neq j}\frac{y_i}{P_i}\cdot (-P_iP_j) =-y_j + y_jP_j +\sum_{i\neq j}y_iP_j= P_j-y_j
  $$

  即为Softmax的结果与其对应标签的独热函数相减。

## Exercise4. 双层神经网络

In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

- Architecture: input - fully connected layer - ReLU - fully connected layer - softmax
- Debug on training:
  1. Plot the loss function and the accuracies on the training and validation sets during optimization.
  2. Visualize the weights that were learned in the first layer of the network.
  3. 损失函数非规律性降低 -> 学习率太低
  4. 训练集和验证集准确率差异不大，但测试表现不好 -> 模型参数量需增大
  5. 调节以下超参数：隐藏层参数量、学习率、训练轮数、参数正则化强度、学习率下降强度等
