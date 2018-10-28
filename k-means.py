import numpy as np
import matplotlib.pyplot as plt
import math

k = 2  # k-means中所设置的k
means = 2  # 生成数据类
m = 2  # 数据属性个数
num = 10  # 各类数据生成个数


# 生成X数据集
# loc:各属性均值，为列表  lable:生成数据的标签   scale:方差  num:生成数据个数    m:生成数据的维度
def birthX(label, loc, scale, num, m):
    X = []
    for i in range(num):
        if(len(X) == 0):
            X = loc+np.random.normal(0, scale, m)
            continue
        X = np.vstack((X, loc+np.random.normal(0, scale, m)))
    return X, np.linspace(label, label, num)


def kmeans(X, k):

    return 0


def draw(C, k):
    plt.subplot(121)
    for i in range(k):
        X = C[i].T
        plt.scatter(X.tolist()[
            0], X.tolist()[1], label="Y = "+str(i))
    plt.show()


loc = [[1, 2], [3, 4]]
scale = [1, 1]
X = []
Y = []
for i in range(means):
    (x, y) = birthX(i, loc[i], scale[i], num, m)
    if(len(X) == 0):
        X = x
        Y = y
        continue
    X = np.vstack((X, x))
    Y = np.vstack((Y, y))
X = np.mat(X)
Y = np.mat(Y)
C = kmeans(X, k)
C = [X[:num], X[num:]]
draw(C, k)
