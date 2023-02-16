import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
# load data
iris = load_iris()

##------------------------------------------------------------
'''
iris的数据集中
iris.data是150x4的矩阵，即150个样本，每个样本均4个特征(花萼长 X^(1); 花萼宽 X^(2); 花瓣长 X^(3); 花瓣宽 X^(4))
iris.target即标签：
                前50个样本的标签为0
                中50个样本的标签为1
                后50个样本的标签为2
之前perceptron和kNN,我们都是拿前100个样本，仅取X^(1)和X^(2)做特征进行的练习
同时由于perceptron要求标签为-1和+1,所以我们对标签有修改

NB这里我们依旧使用iris数据集前100个样本，每个样本取前2个特征，总共2类标签-1(原本是0)/1
'''
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df = df.iloc[:100,[0,1,-1]]
data = np.array(df)
X,y = data[:,:-1], data[:,-1]
y[y==0] = -1
##------------------------------------------------------------

from sklearn.model_selection import train_test_split
# 将数据集和标签集进行划分，30%作为训练集，其余做测试集
test_Size_nB=0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_Size_nB, random_state=2)

import sklearn.naive_bayes as bayes
gNB = bayes.GaussianNB()
bNB = bayes.BernoulliNB()
mNB = bayes.MultinomialNB(alpha=1,fit_prior=True)
catNB = bayes.CategoricalNB()
comNB = bayes.ComplementNB()

gNB.fit(X_train, y_train)
bNB.fit(X_train, y_train)
mNB.fit(X_train, y_train)    #输入数据出现负值,不能使用MultinomialNB
catNB.fit(X_train, y_train)
comNB.fit(X_train, y_train)

# 输出准确率
score_gaussian    = gNB.score(X_test,y_test)
score_bernoulli   = bNB.score(X_test,y_test)
score_multionmial = mNB.score(X_test,y_test)
score_categorical = catNB.score(X_test, y_test)
score_complement  = comNB.score(X_test, y_test)

print("gaussian score:"   +str(score_gaussian))
print("bernoulli score:"  +str(score_bernoulli))
print("multionmial score:"+str(score_multionmial))
print("categorical score:"+str(score_categorical))
print("complement score:" +str(score_complement))

plt.figure()

for fig_loc, nb_type in zip([231,232,233,234,235,236], [gNB, bNB, mNB, catNB, comNB, None]):
    if nb_type is not None:
        nb_predict = nb_type.predict(X_test)
        plt.subplot(fig_loc)
        plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],  c='r',alpha=0.5);
        plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1],c='white',  edgecolors='r',s=60);
        plt.scatter(X_test[nb_predict==1,0], X_test[nb_predict==1,1], c='green', edgecolors='k',label=f'{nb_type} predict-(+1)');

        
        plt.scatter(X_train[y_train==-1,0],X_train[y_train==-1,1],c='b',alpha=0.5);
        plt.scatter(X_test[y_test==-1,0],X_test[y_test==-1,1],c='white',edgecolors='b',s=60);
        plt.scatter(X_test[nb_predict==-1,0],X_test[nb_predict==-1,1],c='orange',edgecolors='k',label=f'{nb_type} predict-(-1)\n score={nb_type.score(X_test,y_test)}');
        plt.grid()
        plt.legend(loc='best')
    else:
        plt.subplot(fig_loc)
        plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],  c='r',label='train-(+1)',alpha=0.5);
        plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1],c='white',  edgecolors='r',s=60,label='test-(+1)');
        plt.scatter(X_train[y_train==-1,0],X_train[y_train==-1,1],c='b',label='train-(-1)',alpha=0.5);
        plt.scatter(X_test[y_test==-1,0],X_test[y_test==-1,1],c='white',edgecolors='b',s=60,label='test-(-1)');
        plt.grid()
        plt.legend()

plt.title(f'Sample size = {test_Size_nB}')
plt.show()

