import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

class PerceptronModel():
    def __init__(self, X, y, eta, iniRand,max_iteration=2000):
        self.feature_size = X.shape[-1] # 确定输入的特征维度,x={x^(1),x^(2)},即维度=2
        self.sample_Num = X.shape[0] # {x1={x1^(1), x1^(2)}, x2={x2^(1), x2^(2)}, x3={x3^(1), x3^(2)}}, 样本数=3
        ## 感知机中，权重向量w的尺寸=特征维度x1
        ##          对偶参数a的尺寸=样本数x1
        ##          Gram matrix的尺寸=样本数x样本数
        if iniRand:
            self.w = np.random.randn(self.feature_size)
            self.b = np.random.randn(1)
        else:
            self.w = np.zeros(self.feature_size)
            self.b = 0
        self.eta = eta # 学习率
        self.dataX = X # 数据
        self.datay = y # 标签
        self.epoch = 0 # 迭代次数
        self.max_iteration = max_iteration # 迭代上限

        # 对偶形式参数
        self.a = np.zeros(self.sample_Num)
        self.GramMatrix = np.zeros((self.sample_Num, self.sample_Num))
        self.calculateGmatrix()

    def sign0(self, x, w, b):  # 原始形式的sign
        return np.dot(w,x)+b
    def sign1(self, a, G_ij, Y, b): # 对偶形式的sign
        # a是simple_Numx1, Y是simple_Numx1, a*Y就是对应元素相乘，还是simple_Numx1；
        # G_ij是simple_Numx1
        # ((a*Y)@G_ij)是标量
        return (a*Y)@G_ij + b 
    def OriginClassifier(self): # 原始形式的感知机
        self.epoch = 0
        updates = 1
        while updates >0 and self.epoch < self.max_iteration:
            updates = 0
            find_1 = False # 遍历训练集时，找到第一个误分点，更新参数后就不再继续遍历当前序列了
            for i in range(len(self.dataX)):
                if not find_1:
                    X = self.dataX[i]
                    y = self.datay[i]
                    if (y*self.sign0(X, self.w, self.b)) <= 0:
                        self.w += self.eta*np.dot(X,y)
                        self.b += self.eta*y
                        updates += 1
                        find_1 = True
            self.epoch+=1
        print("Origin, done, eta=%.2f, total epoch=%d" %(self.eta, self.epoch))
    def calculateGmatrix(self):
        for i in range(self.sample_Num):
            for j in range(0, i+1):
                self.GramMatrix[i][j] = self.dataX[i]@self.dataX[j]
                self.GramMatrix[j][i] = self.GramMatrix[i][j]
    def DualFormClassifier(self):
        self.epoch = 0
        updates = 1
        while updates >0 and self.epoch < self.max_iteration:
            updates = 0
            find_1 = False # 遍历训练集时，找到第一个误分点，更新参数后就不再继续遍历当前序列了
            for i in range(len(self.dataX)):
                if not find_1:
                    y = self.datay[i]
                    G_ij = self.GramMatrix[i]  # 即取GramMatrix的每一列，等于self.GramMatrix[:,i]
                    if (y*self.sign1(self.a, G_ij, self.datay, self.b)) <= 0:
                        self.a[i] += self.eta
                        self.b    += self.eta*y
                        updates +=1
                        find_1 = True
            self.epoch+=1
        print("Dual, done, eta=%.2f, total epoch=%d" %(self.eta, self.epoch))


if __name__ == '__main__':
    # 读取鸢尾花数据
    iris = load_iris()
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df = df.iloc[:100,[0,1,-1]]
    data = np.array(df)
    X,y = data[:,:-1], data[:,-1]
    y[y==0] = -1
    
    plt.scatter(df.iloc[:50,0], df.iloc[:50,1], label='-1')
    plt.scatter(df.iloc[50:100,0], df.iloc[50:100,1], label='1')
    
    # 调用感知机进行分类，学习率eta
    p1 = PerceptronModel(X, y, eta=0.05, iniRand=True, max_iteration=300000000)
    p1.OriginClassifier()  # 原始形式分类

    # # 绘制原始算法分类超平面
    x_points = np.linspace(4, 7, 10)
    y0 = -(p1.w[0] * x_points + p1.b) / p1.w[1]
    plt.plot(x_points, y0, 'r', label='原始算法分类线')

    p1.DualFormClassifier()
    omega0 = sum(p1.a[i] * y[i] * X[i][0] for i in range(len(X)))
    omega1 = sum(p1.a[i] * y[i] * X[i][1] for i in range(len(X)))
    y1 = -(omega0 * x_points + p1.b) / omega1

    # # 绘制对偶算法分类超平面
    plt.plot(x_points, y1, 'b', lw=2, label='对偶算法分类线')

    plt.rcParams['font.sans-serif'] = 'SimHei'  # 消除中文乱码
    plt.grid()
    plt.legend()
    plt.show()