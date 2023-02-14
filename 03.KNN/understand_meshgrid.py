'''
https://blog.csdn.net/lllxxq141592654/article/details/81532855
对meshgrid的讲解十分清晰
用meshgrid就是为了生成2个矩阵，X,Y
X中每个元素Xij就是对应点的横坐标
Y中每个元素Yij就是对应点的纵坐标
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([0, 1])

X, Y = np.meshgrid(x, y)
print(X)
print(Y)


plt.plot(X, Y,
         color='red',  # 全部点设置为红色
         marker='.',  # 点的形状为圆点
         linestyle='')  # 线型为空，也即点与点之间不用线连接
plt.grid(True)
plt.show()