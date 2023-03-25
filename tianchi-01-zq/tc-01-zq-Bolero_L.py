#/ 
# @Author: Fei Chen
# @Date: 2023-03-14 16:07:21
# @LastEditTime: 2023-03-14 16:23:32
# @LastEditors: Fei Chen
# @Description: 
#/
# 导库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

#  数据读取
train_data = pd.read_table('./zhengqi_train.txt')
test_data = pd.read_table('./zhengqi_test.txt')
print(train_data.describe())
# 3.1 检查有无NULL
print(train_data.isnull().sum())
# 3.2 数据划分
train_data_X = train_data.drop(['target'], axis = 1)
train_data_y = train_data['target']
# 3.3 比较特征变量
# plt.figure(figsize=(30,30))
# i = 1
# for col in test_data.columns:
#     plt.subplot(5,8,i)
#     sns.distplot(train_data_X[col], color = 'red')
#     sns.distplot(test_data[col], color = 'blue')
#     plt.legend(['Train', 'Test'])   
#     i += 1
# plt.show()
# 3.4 删除差异较大变量
train_data_X_new = train_data_X.drop(['V2','V5','V9','V11','V13','V14','V17','V19','V20','V21','V22','V27'], axis = 1)
test_data_new = test_data.drop(['V2','V5','V9','V11','V13','V14','V17','V19','V20','V21','V22','V27'], axis = 1)
all_data_X = pd.concat([train_data_X_new,test_data_new])
# 3.5 数据集的切割
X_train, X_test, y_train, y_test = train_test_split(train_data_X_new, train_data_y, test_size = 0.3, random_state = 827)

'''
4.1 Linear Regression
4.2 Linear SVR
4.3 RandomForest Regression
4.4 XGB Regression
'''

# 4.1 线性回归
def Linear_Regression(X_train,X_test,y_train,y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    mse = mean_squared_error(y_test,model.predict(X_test))

    print('Linear_Regression的训练集得分：{}'.format(model.score(X_train,y_train)))
    print('Linear_Regression的测试集得分：{}'.format(model.score(X_test,y_test)))
    print('Linear_Regression的测试集的MSE得分为：{}'.format(mse))
    print('--------------------------------')

# 4.2 SVM
def Linear_SVR(X_train,X_test,y_train,y_test):
    model = LinearSVR()
    model.fit(X_train,y_train)
    mse = mean_squared_error(y_test,model.predict(X_test))
    
    print('Linear_SVR的训练集得分：{}'.format(model.score(X_train,y_train)))
    print('Linear_SVR的测试集得分：{}'.format(model.score(X_test,y_test)))
    print('Linear_SVR的测试集的MSE得分为：{}'.format(mse))
    print('--------------------------------')

# 4.3 随机森林
def RandomForest_Regressor(X_train,X_test,y_train,y_test,n_estimators = 70):
    model = RandomForestRegressor(n_estimators= n_estimators)
    model.fit(X_train,y_train)
    mse = mean_squared_error(y_test,model.predict(X_test))
    print('RandomForest_Regressor的训练集得分：{}'.format(model.score(X_train,y_train)))
    print('RandomForest_Regressor的测试集得分：{}'.format(model.score(X_test,y_test)))
    print('RandomForest_Regressor的测试集的MSE得分为：{}'.format(mse))
    print('--------------------------------') 

# XGBRegression
def XGB_Regressor(X_train,X_test,y_train,y_test):
    model = XGBRegressor(objective ='reg:squarederror')
    model.fit(X_train,y_train)
    mse = mean_squared_error(y_test,model.predict(X_test))
    print('XGB_Regressor的训练集得分：{}'.format(model.score(X_train,y_train)))
    print('XGB_Regressor的测试集得分：{}'.format(model.score(X_test,y_test)))
    print('XGB_Regressor的测试集的MSE得分为：{}'.format(mse))
    print('--------------------------------')


Linear_Regression(X_train, X_test, y_train, y_test)
Linear_SVR(X_train, X_test, y_train, y_test)
RandomForest_Regressor(X_train, X_test, y_train, y_test, n_estimators = 70)
XGB_Regressor(X_train, X_test, y_train, y_test)

'''
由4个模型的mse得分可知，RandomForest和XGB的mse更低，即模型效果更好
所以下面调参，只针对RandomForest和XGB进行
'''

# 得出每个模型的predict
# lr=LinearRegression()
# lr.fit(X_train,y_train)
# linear_predict = lr.predict(X_test)

# svr=LinearSVR()
# svr.fit(X_train,y_train)
# svr_predict = svr.predict(X_test)

# rf=RandomForestRegressor(n_estimators = 200, max_features = 'sqrt')
# rf.fit(X_train,y_train)
# rf_predict = rf.predict(X_test)

# xgb=XGBRegressor(learning_rate = 0.1, 
#                 n_estimators = 200, 
#                 max_depth = 6, 
#                 min_child_weight =9, 
#                 seed = 0,
#                 subsample = 0.8, 
#                 colsample_bytree = 0.8, 
#                 gamma = 0.3, 
#                 reg_alpha = 0, 
#                 reg_lambda = 1,
#                 objective ='reg:squarederror')
# xgb.fit(X_train,y_train)
# xgb_predict = xgb.predict(X_test)

