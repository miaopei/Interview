Index
---
<!-- TOC -->

- [1. 梯度爆炸的解决办法***](#梯度爆炸的解决办法)
- [14. 批标准化（Batch Normalization）的意义**](#批标准化(Batch Normalization)的意义)

<!-- /TOC -->

## 1. 梯度爆炸的解决办法***

1. 梯度截断（gradient clipping）——如果梯度超过某个阈值，就对其进行限制

2. 良好的参数初始化策略也能缓解梯度爆炸问题（权重正则化）

3. 使用线性整流激活函数，如 ReLU 等

## 2. 神经网络（MLP）的万能近似定理*

一个前馈神经网络如果具有至少一个非线性输出层，那么只要给予网络足够数量的隐藏单元，它就可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的函数。

## 3. 神经网络中，深度与宽度的关系，及其表示能力的差异**

隐藏层的数量称为模型的深度，隐藏层的维数（单元数）称为该层的宽度。
万能近似定理表明一个单层的网络就足以表达任意函数，但是该层的维数可能非常大，且几乎没有泛化能力；此时，使用更深的模型能够减少所需的单元数，同时增强泛化能力（减少泛化误差）。参数数量相同的情况下，浅层网络比深层网络更容易过拟合。

## 4. 在深度神经网络中，引入了隐藏层（非线性单元），放弃了训练问题的凸性，其意义何在？**

放弃训练问题的凸性，简单来说，就是放弃寻求问题的最优解。

非线性单元的加入，使训练问题不再是一个凸优化问题。这意味着神经网络很难得到最优解，即使一个只有两层和三个节点的简单神经网络，其训练优化问题仍然是 NP-hard 问题 (Blum & Rivest, 1993).

但即使如此，使用神经网络也是利大于弊的：

- 人类设计者只需要寻找正确的函数族即可，而不需要去寻找精确的函数。
- 使用简单的梯度下降优化方法就可以高效地找到足够好的局部最小值
- 增强了模型的学习/拟合能力，如原书中所说“ maxout 单元可以以任意精度近似任何凸函数”。至于放弃凸性后的优化问题可以在结合工程实践来不断改进。 “似乎传统的优化理论结果是残酷的，但我们可以通过工程方法和数学技巧来尽量规避这些问题，例如启发式方法、增加更多的机器和使用新的硬件（如GPU）。”

## 5. 为什么交叉熵损失相比均方误差损失能提高以 sigmoid 和 softmax 作为激活函数的层的性能？**

简单来说，就是使用均方误差（MSE）作为损失函数时，会导致大部分情况下梯度偏小，其结果就是权重的更新很慢，且容易造成“梯度消失”现象。而交叉熵损失克服了这个缺点，当误差大的时候，权重更新就快，当误差小的时候，权重的更新才慢。

具体推导过程如下：

> https://blog.csdn.net/guoyunfei20/article/details/78247263 - CSDN 博客

> 这里给出了一个具体的[例子](https://blog.csdn.net/shmily_skx/article/details/53053870)

## 6. 分段线性单元（如 ReLU）代替 sigmoid 的利弊***

- 当神经网络比较小时，sigmoid 表现更好；

- 在深度学习早期，人们认为应该避免具有不可导点的激活函数，而 ReLU 不是全程可导/可微的

- sigmoid 和 tanh 的输出是有界的，适合作为下一层的输入，以及整个网络的输出。实际上，目前大多数网络的输出层依然使用的 sigmoid（单输出） 或 softmax（多输出）。

> 为什么 ReLU 不是全程可微也能用于基于梯度的学习？——虽然 ReLU 在 0 点不可导，但是它依然存在左导数和右导数，只是它们不相等（相等的话就可导了），于是在实现时通常会返回左导数或右导数的其中一个，而不是报告一个导数不存在的错误。

> > 一阶函数：可微==可导

- 对于小数据集，使用整流非线性甚至比学习隐藏层的权重值更加重要 (Jarrett et al., 2009b)

- 当数据增多时，在深度整流网络中的学习比在激活函数具有曲率或两侧饱和的深度网络中的学习更容易 (Glorot et al., 2011a)：传统的 sigmoid 函数，由于两端饱和，在传播过程中容易丢弃信息

- ReLU 的过程更接近生物神经元的作用过程

> 饱和（saturate）现象：在函数图像上表现为变得很平，对输入的微小改变会变得不敏感。

> https://blog.csdn.net/code_lr/article/details/51836153 - CSDN博客

> > 答案总结自该知乎问题：https://www.zhihu.com/question/29021768

## 7. 在做正则化过程中，为什么只对权重做正则惩罚，而不对偏置做权重惩罚*

在神经网络中，参数包括每一层仿射变换的权重和偏置，我们通常只对权重做惩罚而不对偏置做正则惩罚。

精确拟合偏置所需的数据通常比拟合权重少得多。每个权重会指定两个变量如何相互作用。我们需要在各种条件下观察这两个变量才能良好地拟合权重。而每个偏置仅控制一个单变量。这意味着，我们不对其进行正则化也不会导致太大的方差。另外，正则化偏置参数可能会导致明显的欠拟合。

## 8. 列举常见的一些范数及其应用场景，如 L0、L1、L2、L∞、Frobenius等范数**

$L_0$：向量中非零向量的个数

$L_1$：向量中所有元素的绝对值之和

$$|x|_1 = \sum_{i}|x_i|$$

$L_2$：向量中所有元素的平方和的开方

$$||x||_2 = \sqrt{\sum_{i}|x_i|^2}$$

其中 L1 和 L2 范数分别是 Lp (p>=1) 范数的特例：

$$||x||_p = (\sum_{i}|x_i|^p)^ \frac{1}{p}$$

**范数的应用**：

- 正则化——权重衰减/参数范数惩罚

**权重衰减的目的**

- 限制模型的学习能力，通过限制参数 θ 的规模（主要是权重 w 的规模，偏置 b 不参与惩罚），使模型偏好于权值较小的目标函数，防止过拟合。

## 9. L1 和 L2 范数的异同***

**相同点**

- 限制模型的学习能力，通过限制参数的规模，使模型偏好于权值较小的目标函数，防止过拟合。

**不同点**

- L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择；一定程度上防止过拟合
- L2 正则化主要用于防止模型过拟合
- L1 适用于特征之间有关联的情况；L2 适用于特征之间没有关联的情况

> [机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975) - CSDN博客

## 10. 简单介绍常用的激活函数，如 sigmoid、relu、softplus、tanh、RBF 及其应用场景***


## 11. 训练误差、泛化误差；过拟合、欠拟合；模型容量，表示容量，有效容量，最优容量的概念； 奥卡姆剃刀原则*

### 11.1 过拟合的一些解决方案***

- 参数范数惩罚（Parameter Norm Penalties）
- 数据增强（Dataset Augmentation）
- 提前终止（Early Stopping）
- 参数绑定与参数共享（Parameter Tying and Parameter Sharing）
- Bagging 和其他集成方法
- Dropout
- 批标准化（Batch Normalization）

## 12. 高斯分布的广泛应用的原因**

高斯分布，即正态分布（normal distribution）：

$$N(x; \mu, \sigma ^2) = \sqrt{\frac{1}{2 \pi \sigma ^2}} exp(- \frac{1}{2 \sigma ^2}(x - \mu)^2)$$

概率密度函数图像：

<img src= "source/高斯密度分布.png">

其中峰的 `x` 坐标由 `µ` 给出，峰的宽度受 `σ` 控制；特别的，当 `µ = 0, σ = 1` 时，称为标准正态分布

正态分布的均值 `E = µ`；标准差 `std = σ`，方差为其平方

## 13. Dropout 与 Bagging 集成方法的关系，Dropout 带来的意义与其强大的原因***


## 14. 批标准化（Batch Normalization）的意义**

批标准化（Batch Normalization, BN, Ioffe and Szegedy, 2015）是为了克服神经网络层数加深导致难以训练而出现的一个算法。

说到底，BN 还是为了解决梯度消失/梯度爆炸问题，特别是梯度消失。

BN 算法：

<img src="source/BN算法.png">

BN 算法需要学习两个参数 γ 和 β.

> Ioffe and Szegedy, 2015, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**批标准化为什么有用？**

> [深度学习（二十九）Batch Normalization 学习笔记](https://blog.csdn.net/hjimce/article/details/50866313) - CSDN博客

> [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762/answer/85238569) - 知乎

























数学公式：

<div align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" title="ax^{2} + by^{2} + c = 0" /></a></div>

<a href="https://www.codecogs.com/eqnedit.php?latex=ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" title="ax^{2} + by^{2} + c = 0" /></a>

## Reference

### 数学公式html生成器

- [在线 LaTeX 编辑器](https://www.codecogs.com/latex/eqneditor.php)

### Interview Link

- [那些深度学习《面试》你可能需要知道的](https://zhuanlan.zhihu.com/p/29965072) - 知乎



### 机器学习

- [详解机器学习中的梯度消失、爆炸原因及其解决方法](https://blog.csdn.net/qq_25737169/article/details/78847691) - CSDN
- [机器学习中防止过拟合的处理方法](https://blog.csdn.net/heyongluoyao8/article/details/49429629) - CSDN
- [Jacobian 和 Hessian 矩阵及其在深度学习中的重要性](http://jacoxu.com/jacobian%E7%9F%A9%E9%98%B5%E5%92%8Chessian%E7%9F%A9%E9%98%B5/) - 刻骨铭心Blog
- [【机器学习】信息量，信息熵，交叉熵，KL散度和互信息（信息增益）](https://blog.csdn.net/haolexiao/article/details/70142571) - CSDN
- [详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解](https://blog.csdn.net/u011508640/article/details/72815981) - CSDN
- []()

### 深度学习

- [【深度学习】深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html) - 博客园
- [RNN, LSTM, GRU 公式总结](https://blog.csdn.net/zhangxb35/article/details/70060295) - CSDN
- [人工神经网络中的activation function的作用具体是什么？为什么ReLu要好过于tanh和sigmoid function?](https://blog.csdn.net/code_lr/article/details/51836153) - CSDN
- [DeepLearning学习笔记——无监督学习算法](https://blog.csdn.net/hustqb/article/details/77234949) - CSDN
- [深度学习之概述(Overview)](https://juejin.im/entry/59ddf2006fb9a04509087bcf) - 掘金

### Python

- [2018年最常见的Python面试题&答案（上篇）](https://juejin.im/post/5b6bc1d16fb9a04f9c43edc3)
- []()
- []()

### C++











