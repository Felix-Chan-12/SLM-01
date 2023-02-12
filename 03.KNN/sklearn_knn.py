import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df = df.iloc[:100,[0,1,-1]]
data = np.array(df)
X,y = data[:,:-1], data[:,-1]
y[y==0] = -1

from sklearn.model_selection import train_test_split
# 将数据集和标签集进行划分，20%作为训练集，其余做测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors=3,  algorithm='kd_tree', p=2, metric='minkowski')
# n_neighbors=3, KNN的k=3
# algorithm ={‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# p=Power parameter for the Minkowski metric, 即范数指标，Lp的p
# metric='minkowski'，Metric to use for distance computation


plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],c='r',label='train-(+1)');
plt.scatter(X_train[y_train==-1,0],X_train[y_train==-1,1],c='b',label='train-(-1)');


plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1],c='white',edgecolors='r',label='test-(+1)');
plt.scatter(X_test[y_test==-1,0],X_test[y_test==-1,1],c='white',edgecolors='b',label='test-(-1)');

knn1.fit(X_train, y_train)
print(knn1.score(X_test, y_test))

# test_p = np.array([[6,3],[5,3]])
# test_p = np.array([[6,3],[4.5,2.8],[5.6,5.6],[3.5,3],[6.5,5]])
test_p = np.random.normal(3, 1, size=120).reshape(60,2)
print(knn1.predict(test_p))

plt.scatter(test_p[knn1.predict(test_p)==+1,0],test_p[knn1.predict(test_p)==+1,1],c='green',edgecolors='k',label=f'score={knn1.score(X_test, y_test)}\n predict=1')
plt.scatter(test_p[knn1.predict(test_p)==-1,0],test_p[knn1.predict(test_p)==-1,1],c='orange',edgecolors='k',label='predict=-1')

plt.legend()
plt.grid()
plt.show()
