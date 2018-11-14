import numpy as np
import matplotlib.pyplot as plt
import math
import sys

k = 3  # k-means中所设置的k
means = 3  # 生成数据类
m = 2  # 数据属性个数
num = 50  # 各类数据生成个数
loc = [[0,0],[0, 2], [10,0]]
scale = [1,1,2]
exitloop = 0.01
maxloop = 50


# 生成X数据集
# loc:各属性均值，为列表  lable:生成数据的标签   scale:方差  num:生成数据个数    m:生成数据的维度
def birthX(label, loc, scale, num, m):
    X = []
    for i in range(num):
        if(len(X) == 0):
            X = loc+np.random.normal(0, scale, m)
            continue
        X = np.vstack((X, loc+np.random.normal(0, scale, m)))
    return X, np.array([label]*num)


#随机生成初始化中心点
def initCenter(X,k):
    center = []
    for i in range(m):
        x = X.T.tolist()
        Max = max(x[i])
        Min = min(x[i])
        dot = np.random.uniform(Min,Max,k)
        center.append(dot.tolist())
    return np.mat(center).T

def distance(x,c):
    return (x-c).dot((x-c).T)[0][0]



def initC(X, average,k):
    mindistance = sys.maxsize
    C = []
    for i in range(k):
        C.append([])
    for x in X:
        min = sys.maxsize
        belong = -1
        for i in range(k):
            c = np.mat(average[i])
            dis = distance(x,c)
            if  (dis<min):
                min = dis
                belong = i
        C[belong].append(x.tolist()[0])
    C = [np.mat(c) for c in C]
    return C

#计算高斯分布概率
#X:1*m  avergae:1*m sigmod:1*m
#返回：1*1
def normalPossibility(X,average,covariance):
    X = np.mat(X)
    index = -((X-average).dot(covariance.I).dot((X-average).T))/2
    return np.power(math.e,index)/math.sqrt(np.power(2*math.pi,X.shape[1])
            *math.fabs(np.linalg.det(covariance)))


#计算协方差
def Covariance(C,center):
    variance = []
    for i in range(len(C)):
        c = C[i]
        sum = [0]*len(center[i])
        for j in range(len(c)):
            sum = (c[j]-center[i]).dot((c[j]-center[i]).T)
        variance.append(np.multiply((np.mat(sum)/len(C[i])),(np.eye(np.mat(C[i]).shape[1]))))
    return variance


#average:各类的均值k*m covraince:各类的协方差矩阵 k*(m*m)  X:num*m     pi:概率 k*1
#返回:num*k
def E_step(average,covraince,X,pi):
    P = [0]*X.shape[0]
    for i in range(X.shape[0]):
        top = []
        for j in range(average.shape[0]):
            T = pi[j]*normalPossibility(X[i],average[j],covariance[j])[0,0]
            top.append(T)
        down = np.sum(top).tolist()
        P[i]=(np.mat(top)/down).tolist()[0]
    return P


#计算似然：average:k*m  covariance:k*(m*m)  pi:k*m
def likelihood(X,average,covariance,pi):
    sum = 0
    for i in range(X.shape[0]):
        t = 0
        for j in range (average.shape[0]):
            t+=pi[j]*normalPossibility(X.tolist()[i],average,covariance[j]).tolist()[0][0]
        sum += math.log(t)
    return sum


def M_step(X,average,covariance,pi):
    oldLikelihood = likelihood(X,average,covariance,pi)
    for iter in range(maxloop):
        gamma = np.mat(E_step(average,covariance,X,pi)).T#k*num
        N = np.sum(gamma.T,axis = 0).tolist()[0]#1*k
        for i in range(len(average)):
            covariance[i]=np.mat([[0]*covariance[i].shape[0]]*covariance[i].shape[0])
            average[i]=gamma[i].dot(X)/N[i]
            for j in range(X.shape[0]):
                c = X[j]-average[i]
                covariance[i]=covariance[i]+np.mat((gamma[i,j]*c.T).dot(c))/N[i]
        pi = (np.mat(N)/X.shape[0]).tolist()[0]
        newLikelihood = likelihood(X,average,covariance,pi)
        if(math.fabs(oldLikelihood-newLikelihood)<exitloop):
            break
        oldLikelihood = newLikelihood
    return average,covariance,pi


def split(X,average,covariance,pi):
    gamma = np.mat(E_step(average,covariance,X,pi))
    C = []
    for i in range(k):
        C.append([])
    for i in range(gamma.shape[0]):
        max = -1
        Max = 0
        for j in range(gamma.shape[1]):
            if(gamma[i,j]>Max):
                max = j
                Max = gamma[i,j]
        C[max].append(X[i].tolist()[0])
    C = [np.mat(c) for c in C]
    return C



def draw(X,C,average, k,m):
    plt.subplot(121)
    plt.title("Data")
    for i in range(means):
        x = X[i*num:(i+1)*num].T
        plt.scatter(x.tolist()[
            0], x.tolist()[1], label="Y = "+str(i))
    L = np.mat(loc).T.tolist()
    plt.scatter(L[0],L[1],label="center",marker="x")
    plt.legend()
    plt.subplot(122)
    plt.title("The Result by EM")
    for i in range(k):
        x = C[i].T
        plt.scatter(x.tolist()[
            0], x.tolist()[1], label="Y = "+str(i))
    average = np.mat(average).T.tolist()
    plt.scatter(average[0],average[1],label="center",marker="x")
    plt.legend()
    plt.show()



X = []
for i in range(means):
    (x, y) = birthX(i, loc[i], scale[i], num, m)
    if(len(X) == 0):
        X = x
        continue
    X = np.vstack((X, x))
X = np.mat(X)
average = initCenter(X,k).tolist()
C = initC(X,average,k)
covariance = Covariance(C,average)
(average,covariance,pi)=M_step(X,np.mat(average),covariance,[1/k]*k)
#EM算法聚类结果
print(average)
C = split(X,average,covariance,pi)
draw(X,C,average,k,m)
