#!/usr/bin/python 
'''
decision tree base on ID3
'''
import numpy as np
import math
import copy

def cal_entrophy(dataset):
    values = dataset[:, -1].T.tolist()[0]
    val_cat = set(values)
    count, attrs = dataset.shape
    entro = 0
    for v in val_cat:
        nums = np.sum(np.equal(values, v).astype(int))
        pro = nums / count 
        entro -= pro * math.log(pro, 2)
    return entro
    
def splitDataset(dataset, attr_index, value):
    new_ds = None
    for d in dataset:
        if d[0,attr_index] == value:
            if new_ds is None:
                new_ds = d
            else:
                new_ds = np.row_stack((new_ds, d))
    return new_ds

def getBestSplitAttr(dataset, attr_list):
    total_entrophy = cal_entrophy(dataset)
    max_gain = -1
    best_attr = -1
    value_category = None
    value_list = None
    for attr in attr_list:
        vlist = dataset[:, attr].T.tolist()[0]
        values = set(vlist)
        entrophy = 0
        for v in values:
            sd = splitDataset(dataset, attr, v)
            entrophy += cal_entrophy(sd)
        gain = total_entrophy - entrophy
        if gain > max_gain:
            best_attr = attr
            max_gain = gain
            value_category = values
            value_list = vlist
    return best_attr, value_category, value_list

def id3_build_tree(dataset, attrs):
    tree = {"root":None, "branches":{}}
    
    if len(attrs) == 0:
        val_list = dataset[:, -1].T.tolist()[0]
        val_cat = set(val_list)
        for v in val_cat:
            tree['branches'][v] = np.sum(np.equal(val_list, v))
            return tree
            
    best_attr, val_cat, vlist = getBestSplitAttr(dataset, attrs)
    if len(val_cat) == 1:
        v = val_cat.pop()
        tree["branches"][v] = len(vlist)
        return tree
        
    tree["root"] = best_attr
    attrs.pop(attrs.index(best_attr))    
    for v in val_cat:
        subd = splitDataset(dataset, best_attr, v)
        tree['branches'][v] = id3_build_tree(subd, copy.copy(attrs))
    return tree
if __name__ == '__main__':
    data = [
        [1, 1, 1]
        ,[1, 1, 1]
        ,[1, 0, 0]
        ,[0, 1, 0]
        ,[0, 1, 0]
    ]
    data = np.matrix(data)
    print(id3_build_tree(data, [0, 1])) 
    
