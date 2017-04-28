# The Machine Learing Training in HIT

作业的发布与提交都在本git仓库中进行

### 相关资源：

* 机器学习视频网站--**请与课程的进度保持同步**。http://www.coursera.org/learn/machine-learning/
* matlab下载地址--windows & linux。链接: https://pan.baidu.com/s/1o787KeA 密码: ad6a
* 数据挖掘导论--作为补充材料。发布于群文件
* python极简入门--http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000#0
* python书籍推荐--Python编程(programming python)
* numpy教程--https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
* matplotlib教程--http://www.labri.fr/perso/nrougier/teaching/matplotlib/
* 矩阵求导--https://zhuanlan.zhihu.com/p/24709748
  https://ccjou.wordpress.com/2013/05/31/%E7%9F%A9%E9%99%A3%E5%B0%8E%E6%95%B8/
* Desmos图形计算器--https://www.desmos.com/calculator

### 欢迎来到新手村，你需要完成以下训练：

#### 1. Task0——github的使用

1. 注册github帐号
2. star这个仓库
3. 自学git及github的基础使用，并在自己电脑上配置好环境(git教程http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
4. fork这个仓库，clone下你的仓库，在本地创建新的分支（不要使用默认的master分支）
5. 在本地Task0目录下，添加一个新的文件，以自己的github账户名作为markdown文件的文件名，如：Red-Night-Aria.md
6. 文件中写上自己的本名+你想说的任何话 （比如自己的座右铭）
7. commit到本地仓库，然后将本地分支推送到remote
8. 向原仓库的master分支发起pull request，等待合并分支

**请于第四周结束前完成以上内容。**



#### 2. Task1——Linear Regression

1. 加载数据Task1/q2x.dat，Task1/q2y.dat
2. 使用线性回归算法拟合它们
3. 画图观察梯度下降的过程,let it nice-looking:)。
4. 结果图参考Task1/linear_regress.fig
5. 在Task1目录下提交你的结果图(jpg、png、fig)与代码，以自己的github账户名作为文件名，如：Red-Night-Aria.fig

##### 支线任务：
* 学习python的基本语法
* 学习numpy框架的使用
* 使用python完成Task1

**请于第六周结束前完成以上内容。**

#### 3. Task2——Locally Weighted Linear Regression

1. 加载数据Task2/q2x.dat，Task2/q2y.dat
2. 使用加权线性回归算法拟合它们(请自行查阅局部加权线性回归的资料)  
3. 改变高斯函数的theta值，观察结果
4. 画图，结果图参考Task2/Red-Night-Aria.png
5. 在Task2目录下提交你的结果图(jpg、png、fig)与代码，以自己的github账户名作为文件名，如：Red-Night-Aria.fig

**请于第七周结束前完成以上内容。**

#### 4. Task3——Logistic Regression

1. 加载数据Task3/task3\_x.dat，Task3/task3\_y.dat
2. 使用逻辑斯底回归算法对它们进行分类
3. 结果图参考Task3/Red-Night-Aria.png
4. 在Task3目录下提交你的结果图(jpg、png)，以自己的github账户名作为文件名。

##### 支线任务：
* 用牛顿法取代梯度下降法实现算法

**请于第八周结束前完成以上内容。**

#### 5. Task4——Simple Neural Network(part 1)

1. 加载数据Task4/watermelon.txt
2. 使用单隐层神经网络拟合对数据进行分类
3. 结果图参考Task4/Red-Night-Aria.png
4. 在Task4目录下提交你的结果图(jpg、png、fig)与代码，以自己的github账户名作为文件名。

##### 支线任务：
* 论证以密度和含糖率来分辨西瓜的好坏是否合理。

**请于第九周结束前完成以上内容**

#### 6. Task5——Simple Neural Network(part 2)

概览：学习python的ML工具箱scikit-learn的使用，完成CV界的hello world：MNIST手写体数字识别

0. 安装第三方库scikit-learn
1. 解码MNIST数据集(数据文件的格式见官网http://yann.lecun.com/exdb/mnist/)
2. 使用sklearn库设计神经网络，完成对MNIST数据集的训练(参考文档http://scikit-learn.org/dev/modules/neural_networks_supervised.html#neural-networks-supervised)
3. 利用训练好的模型，预测自己的手写数字图片
4. 在Task5目录下提交你的结果图与代码。(结果图的格式与内容随意 XD

(tips: 训练前用Pillow或matplotlib观察待训练数据，确保数据格式无误)

#### 支线任务：
* 理解MLPClassifier的构造函数中各个参数的意义
* 修改MLPClassifier构造函数的默认参数，使用交叉检验法比较不同模型的性能