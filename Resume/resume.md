1. kaggle猫狗大战项目，使用Keras将Xception， InceptionV3和ResNet50 这三个模型进行迁移学习，在验证集上的准确率可以达到99.4%， 在kaggle提交的结果为0.04150，识别准确率进入 Kaggle Top 2%。项目同时基于训练好的模型使用 Flask 搭建部署了一个简单易用的猫狗识别网页应用。 webapp用daocker打包成了镜像文件，可以快速部署并复现实验结果。
2. 电影推荐系统，使用开源MovieLens数据集，基于TensorFlow使用文本卷积网络生成movie特征矩阵和user特征矩阵，在推荐同类型电影根据当前看的电影特征向量与整个电影特征矩阵的余弦相似度，取相似度最大的Top N个。训练完成后的模型预测 MSE loss 为0.8。 





1. 波士顿房价预测，

