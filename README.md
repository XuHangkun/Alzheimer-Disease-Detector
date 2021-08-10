# PRCV 2021 阿尔茨海默病分类技术挑战赛

## 最优模型介绍

### 数据预处理

* 去掉所有存在无意义值(Nan,inf,-inf)的feature
* 归一化，对每一个feature计算其mean以及std, 然后(value - mean)/std 进行归一化
* 将不同脑谱的数据进行拆分，最后用了AAL,Hammers,rBN这三个脑谱的信息

### 模型
* splitbaseline
* 用了跨越连接从而训练深层模型

### 模型融合

* 对每种图谱的数据，采用10折交叉训练，并且挑选其中的某些折的模型，计算其输出的平均值进行融合
* AAL 第3折 (程序固定了随机数种子，所以每次都相同的数据划分)
* Hammers 第2,3,4,5,7,8,10 折
* rBN 第3折

## 程序运行
* code/script下面有脚本，直接运行就可以训练模型存储在model下
* 可以用code/analysis下面的python脚本查看模型训练的一些信息，loss,f1等
* 最后到code_sub目录下面，先修改mergeAtlas.py脚本,选择你想要融合的模型,然后运行此脚本，生成单一的model.pth
* 修改custum_service.py就可以测试了