
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None)

'''GaussianNB类的主要参数仅有一个，即先验概率priors ，对应Y的各个类别的先验概率P(Y=Ck)。
这个值默认不给出，如果不给出此时P(Y=Ck)=mk/m。其中m为训练集样本总数量，mk为输出为第k类别的训练集样本数。
如果给出的话就以priors 为准。'''

#拟合数据
clf.fit(X, Y)
print("==Predict result by predict==")
print(clf.predict([[-0.8, -1]]))
print("==Predict result by predict_proba==")
print(clf.predict_proba([[-0.8, -1]]))
print("==Predict result by predict_log_proba==")
print(clf.predict_log_proba([[-0.8, -1]]))

# ==Predict result by predict==
# [1]  即[-0.8, -1]的标签为1
# ==Predict result by predict_proba==
# [[9.99999949e-01 5.05653254e-08]]    即[-0.8, -1]的标签为1的概率为9.99999949e-01, 标签为2的概率为5.05653254e-08
# ==Predict result by predict_log_proba==
# [[-5.05653266e-08 -1.67999998e+01]]  即ln(9.99999949e-01)=-5.056e-8, ln(5.05653254e-08)=-1.67999998e+01

'''MultinomialNB(*, alpha=1.0, force_alpha='warn', fit_prior=True, class_prior=None)
<1>参数alpha即为上面的常数λ,λ=1即拉普拉斯平滑，用默认的1即可。如果发现拟合的不好，可以选择稍大于1或者稍小于1的数。
set 'alpha'=0 and 'force_alpha'=True, for no smoothing
<2>布尔参数fit_prior表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率。否则可以用class_prior，
<3>第三个参数class_prior为指定的先验概率
或者不输入第三个参数class_prior让MultinomialNB自己从训练集样本来计算先验概率，此时的先验概率为P(Y=Ck)=mk/m。
其中m为训练集样本总数量，mk为输出为第k类别的训练集样本数。
总结如下：
fit_prior   class_prior          最终先验概率
fals        填或者不填没有意义     P(Y=Ck)=1/k
true        不填                  P(Y=Ck)=mk/m
true        填                    P(Y=Ck)=class_prior'''


'''BernoulliNB(*, alpha=1.0, force_alpha='warn', binarize=0.0, fit_prior=True, class_prior=None)
BernoulliNB一共有4个参数
其中3个参数的名字和意义和MultinomialNB完全相同。
唯一增加的一个参数是binarize。
这个参数主要是用来帮BernoulliNB处理二项分布的，可以是数值或者不输入。
如果不输入，则BernoulliNB认为每个数据特征都已经是二元的。
否则的话，小于binarize的会归为一类，大于binarize的会归为另外一类。
'''

'''ComplementNB 实现了补充朴素贝叶斯(CNB)算法。CNB是标准多项式朴素贝叶斯(MNB)算法的一种自适应算法，特别适用于不平衡的数据集。'''

'''CategoricalNB 对分类分布的数据实现了类别朴素贝叶斯算法。
它假设由索引描述的每个特征都有自己的分类分布'''