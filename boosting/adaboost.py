#!py -3
'''
implement adaboost strategy
'''
import numpy as np

class AdaBoost:
    def __init__(self, weak, weak_trainor, weak_params = {}):
        self.weaks = []
        self.w = weak
        self.wp = weak_params
        self.wt = weak_trainor

    #X: MxN, Y: Mx1
    def train(self, X, Y, max_iterations, acc=0.0):
        wp = self.wp
        cont = X.shape[0] * 1.0
        weights = np.ones((count, 1)/count)
        wp['x'] = X
        wp['Y'] = Y
        for i in range(max_itrations):
            weak_param, error, _Y = self.wt(wp) #_Y:Mx1
            alpha = 0.5 * np.log((1-error)/max(error, 0.001)
            expon = -1 * Y.T * _Y * alpha   #M*1
            weights = weights.T * np.exp(expon) / weights.sum()
            result = alpha * _Y
            correct = (result == Y).astype(float)
            err_rate = correct.sum() / count

            self.weaks.append({
                'alpha':alpha,
                'param':weak_param
            })
            if err_rate <= acc:
                break
    
    def predict(self, X):
        
