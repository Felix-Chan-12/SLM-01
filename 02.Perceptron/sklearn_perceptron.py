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

from sklearn.linear_model import Perceptron

## fit_intercept, bool, default=True Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
## fit_intercept一般就保持默认，即我们认为超平面不一定过原点。如果十分确定S过原点，才设置为False
## eta0即学习率，默认1.0
## tol: float or None, default=1e-3
## The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).
clf = Perceptron(fit_intercept=True, eta0=1, max_iter=1000, shuffle=False, verbose=1, tol=None)
clf.fit(X, y)
w = clf.coef_[0] # e.g., clf.coef_=[[5,10]]; w=[5,10]
print(w)
b = clf.intercept_
print(b)

x_ponits = np.arange(4, 8)
y_ = -(w[0]*x_ponits + b)/w[-1]  # 分母乘过去就是AX+BY+C=0

plt.plot(x_ponits, y_,'r--',label='Hyperplane')
plt.scatter(df.iloc[:50,0], df.iloc[:50,1], label='-1')
plt.scatter(df.iloc[50:100,0], df.iloc[50:100,1], label='1')
plt.grid()
plt.legend()
plt.show()