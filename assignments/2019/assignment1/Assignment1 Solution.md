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
Softmax: L_i = -log(\frac{e^{s_{y_i}}}{\sum_{j}e^{s_j}})
$$
