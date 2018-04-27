#!py -3
'''
implement adaboost strategy
'''
import numpy as np
import random
import pickle
import math

class AdaBoost:
    def __init__(self, weak, weak_trainor, weak_params = {}):
        self.weaks = []
        self.w = weak
        self.wp = weak_params
        self.wt = weak_trainor

    #X: MxN, Y: array (M,)
    def train(self, X, Y, max_iterations, acc=0.0):
        wp = self.wp
        count = X.shape[0]
        weights_result = np.zeros(count)
        weights = np.ones(count, dtype=float) / count              #array (count,)
        for i in range(max_iterations):
            _Y, error, weak_param = self.wt(X, Y, weights, wp) #_Y: array(M,), error:scalar, param:dict
            alpha = 0.5 * np.log((1-error)/max(error, 0.001))
            expon = -1 * Y * _Y * alpha   #array (M,)
            weights = weights * np.exp(expon) / weights.sum()
            weights_result += alpha * _Y
            weights_error = (np.sign(weights_result) != Y).astype(float)
            err_ratio = weights_error.sum() / count

            self.weaks.append({
                'alpha':alpha,
                'param':weak_param
            })
            #print('weights:{}'.format(weights))
            print("iter:{}, Error:{:.3f}".format(i, err_ratio))
            if err_ratio <= acc:
                break
    
    def predict(self, X):
        cls = 0
        for weak in self.weaks:
            cls += self.w(X, weak['param'])
        return np.sign(cls)

    def dump_mode(self):
        with open('./boost.mode', 'wb') as fd:
            pickle.dump(self.weaks, fd, protocol=2)
    
    def load_mode(self, model):
        with open(model, 'rb') as fd:
            self.weaks = pickle.load(fd)
'''
弱分类器及其训练器
'''        
def weak(X, params):
    threshold = params['threshold']
    attr = params['attr']
    op = params['op']
    return np.where(getattr(np, op)(X[:, attr], threshold),-1, 1)

def train_weak(X, Y, weights, params):
    min_loss = 1000000
    best_result = None
    w_param = None
    errir_ratio = None
    batch, attrs = X.shape
    
    numsteps = 300
    for attr in range(attrs):
        maxv = np.max(X[:, attr])
        minv = np.min(X[:, attr])
        step = (maxv - minv) / numsteps
        for j in range(-1, numsteps+1):
            th = minv + j * step
            for op in ['greater','less_equal']:
                wp = {'threshold':th, 'op':op, 'attr':attr}
                result = weak(X, wp)
                error = (result != Y).astype(int)
                loss = (weights * error).sum()
                if loss < min_loss:
                    min_loss = loss
                    best_result = result
                    w_param = wp
                    errir_ratio = np.sum(error) / len(error)

    return best_result, errir_ratio, w_param

if __name__ == '__main__':
    ada = AdaBoost(weak, train_weak, {})
    
    
    X = np.zeros((10000, 36))
    Y = np.zeros(10000)
    fb = open('./features', 'rb')
    fcsv = open('./anno.csv', 'r')
    for i, line in enumerate(fcsv.readlines()):
        label = int(line.strip().split(',')[1])
        Y[i] = (label if label == 1 else -1)
        X[i, :] = (np.reshape(pickle.load(fb), -1))
    fb.close()
    fcsv.close()
    '''
    X = np.reshape([
        [1., 2.]
        ,[2., 1.1]
        ,[1.3, 1.]
        ,[1., 1.]
        ,[2., 1.]
    ],(5,2))
    Y = np.reshape([1., 1., -1., -1., 1.], -1)
    '''
    
    print("****** START TRAINING, GOOD LUCK ******")
    ada.train(X, Y, 100, 0.1)
    print("****** DONE TRAINING, BYE BYE ******")
    ada.dump_mode()
        