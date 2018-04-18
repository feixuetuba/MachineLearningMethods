import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
numpy>=1.12.0
scikit-learn>=0.18.1
使用KNN对鸢尾花数据集进行分类，依赖库：
    sklearn,numpy,scipy
iris数据集说明：
    ris包含150个样本，对应数据集的每行数据。每行数据包含每个样本的四个特征和样本的类别信息，
所以iris数据集是一个150行5列的二维表。
    通俗地说，iris数据集是用来给花做分类的数据集，每个样本包含了花萼长度、花萼宽度、花瓣长度、
花瓣宽度四个特征（前4列），我们需要建立一个分类器，分类器可以通过样本的四个特征来判断样本属于山鸢尾、
变色鸢尾还是维吉尼亚鸢尾（这三个名词都是花的品种）。
    iris的每个样本都包含了品种信息，即目标属性（第5列，也叫target或label）。
    [花萼长度,花萼宽度,花瓣长度,花瓣宽度,属种]
总结下：
    KNN 实现简单，已知数据集必须接近真实数据集，数据集越大运算量越大（必须和每个点算距离），特征选取很重要
'''

class Solver:
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
        
    def classify(self, input, k):
        rols = self.datas.shape[0]
        diffv = self.datas - np.tile(input, [rols, 1])    #把输入向量堆叠成rols行input.shape[1]列的矩阵
        distence = np.sum(pow(diffv, 2), 1)     #按列求和
        sorted_index = distence.argsort()   #获取距离按从小到大排序后的索引值，不改变元数据
        results = {}
        #获取前k个
        for i in range(k):
            lindex = sorted_index[i]
            label = self.labels[lindex]
            if label in results:
                results[label] += 1
            else:
                results[label] = 1
        return sorted(results.items(), key=lambda v:v[1], reverse=True)[0][0]   #出现次数最多的作为结果
  
if __name__ == '__main__':
    import time
    iris = load_iris()
    X = iris.data
    Y = iris.target
    #把数据随机分成测试和训练两个部分， 测试集占1/3
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    solver = Solver(x_train, y_train)
    results = []
    startt = time.time()
    for test in x_test:
        results.append(solver.classify(test, 10))
    print("speed:{:5.4f}s".format((time.time() - startt) / y_test.shape[0]))
    accuracy = accuracy_score(y_true=y_test, y_pred=results)
    print('KNN accuracy : {}'.format(accuracy))