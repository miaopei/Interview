# 1. 逻辑斯蒂回归（Logistic Regression）


## Reference

- [Logistic Regression（逻辑回归）原理及公式推导](https://blog.csdn.net/programmer_wei/article/details/52072939) - CSDN

- [逻辑回归推导](https://www.cnblogs.com/daguankele/p/6549891.html) - 罐装可乐 - 博客园

- [极大似然估计详解](https://blog.csdn.net/zengxiantao1994/article/details/72787849) - CSDN

- [一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750) - 知乎

- [梯度下降原理及Python实现](https://blog.csdn.net/programmer_wei/article/details/51941358) - CSDN

# 2. 支持向量机（SVM）

SVM 和 Logistic 回归的比较：

（1）经典的SVM，直接输出类别，不给出后验概率；

（2）Logistic回归，会给出属于哪一个类别的后验概率；

（3）比较重点是二者目标函数的异同。

TODO

- 凸二次优化
- 拉格朗日乘子法

## Reference

- [SVM中支持向量的通俗解释](https://blog.csdn.net/AerisIceBear/article/details/79588583) - CSDN
- [支持向量机SVM推导及求解过程](https://blog.csdn.net/american199062/article/details/51322852#commentBox) - CSDN

# 3. 决策树

决策树的训练通常由三部分组成：

- 特征选择
- 树的生成
- 剪枝

## 信息增益与信息增益比

TODO

## 分类树 - ID3 决策树与 C4.5 决策树 

ID3 决策树和 C4.5 决策树的区别在于：**前者使用信息增益来进行特征选择，而后者使用信息增益比。**

TODO

## 决策树如何避免过拟合

TODO

## 回归树 - CART 决策树

> 《统计学习方法》 5.5 CART 算法

- CART 算法是在给定输入随机变量 _`X`_ 条件下输出随机变量 _`Y`_ 的**条件概率分布**的学习方法。 
- CART 算法假设决策树是**二叉树**，内部节点特征的取值为“**是**”和“**否**”。

  这样的决策树等价于递归地二分每个特征，**将输入空间/特征空间划分为有限个单元**，然后在这些单元上确定在输入给定的条件下输出的**条件概率分布**。
- CART 决策树**既可以用于分类，也可以用于回归**；

对回归树 CART 算法用**平方误差最小化**准则来选择特征，对分类树用**基尼指数最小化**准则选择特征

## Reference

- []()

# 4. 集成学习

- 基本思想：由多个学习器组合成一个性能更好的学习器
- **集成学习为什么有效？**——不同的模型通常会在测试集上产生不同的误差。平均上，集成模型能至少与其任一成员表现一致；并且**如果成员的误差是独立的**，集成模型将显著地比其成员表现更好。

> 《深度学习》 7.11 Bagging 和其他集成方法

## 集成学习的基本策略

### 1. Boosting

- **Boosting**（提升）方法从某个**基学习器**出发，反复学习，得到一系列基学习器，然后组合它们构成一个强学习器。

- Boosting 基于**串行策略**：基学习器之间存在依赖关系，新的学习器需要依据旧的学习器生成。

- **代表算法/模型**：
  - [提升方法 AdaBoost](#提升方法-adaboost)
  - 提升树
  - 梯度提升树 GBDT

**Boosting 策略要解决的两个基本问题**

1. 每一轮如何改变数据的权值或概率分布？
2. 如何将弱分类器组合成一个强分类器？

### 2. Bagging

- Bagging 基于**并行策略**：基学习器之间不存在依赖关系，可同时生成。

- **代表算法/模型**：
  - [随机森林](#随机森林)
  - 神经网络的 **Dropout** 策略

## AdaBoost 算法

AdaBoost，是英文"Adaptive Boosting"（自适应增强）的缩写。

- [Adaboost算法原理分析和实例+代码（简明易懂）](https://blog.csdn.net/guyuealian/article/details/70995333) - CSDN

## 前向分步算法


## Reference

- [【机器学习】集成学习(三)----前向分步算法、提升树与GBDT](https://blog.csdn.net/u013597931/article/details/79874439) - CSDN

- [梯度提升树(GBDT)原理小结](https://www.cnblogs.com/pinard/p/6140514.html) - 刘建平Pinard

- [GBDT（梯度提升决策树）](https://www.zybuluo.com/evilking/note/946535)

# 5. 梯度提升决策树 GBDT



# 6. 随机森林

随机森林的生成方法：

1. 从样本集中通过重采样的方式产生 n 个样本
2. 假设样本特征数目为 a，对 n 个样本选择 a 中的 k 个特征，用建立决策树的方式获得最佳分割点
3. 重复 m 次，产生 m 棵决策树
4. 多数投票机制来进行预测

**（需要注意的一点是，这里 m 是指循环的次数，n 是指样本的数目，n 个样本构成训练的样本集，而 m 次循环中又会产生 m 个这样的样本集）**

## Reference

- [对于随机森林的通俗理解](https://blog.csdn.net/mao_xiao_feng/article/details/52728164) - CSDN

- [随机森林算法学习(RandomForest)](https://blog.csdn.net/qq547276542/article/details/78304454) - CSDN

- [机器学习中Bagging和Boosting的区别](https://blog.csdn.net/u013709270/article/details/72553282) - CSDN

- [随机森林与决策树](https://clyyuanzi.gitbooks.io/julymlnotes/content/rf.html) - 机器学习笔记

- []()

# 7. 降维

## SVD

## PCA

## Reference

- [降维方法PCA与SVD的联系与区别](https://www.cnblogs.com/bjwu/p/9280492.html) - 彎道超車

- [PCA和SVD降维](https://blog.csdn.net/tianhaoyedl/article/details/77477568) - CSDN

- [数据降维与可视化——t-SNE](https://blog.csdn.net/hustqb/article/details/78144384) - CSDN

- [t-SNE实践——sklearn教程](https://blog.csdn.net/hustqb/article/details/80628721) - CSDN





## Reference

- [使用集成学习提升机器学习算法性能](https://blog.csdn.net/u010099080/article/details/77720711) - CSDN

- [快速理解bootstrap,bagging,boosting-三个概念](https://blog.csdn.net/wangqi880/article/details/49765673) - CSDN

- []()