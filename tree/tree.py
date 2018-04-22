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
    return best_attr, value_category

def id3_build_tree(dataset, attrs):
    tree = {"root":None, "branches":{}}
    val_list = dataset[:, -1].T.tolist()[0]
    val_cat = set(val_list)
    
    if len(val_cat) == 0:
        print(dataset.shape)
    
    if len(attrs) == 0:
        print("att is zero")
        for v in val_cat:
            tree['branches'][v] = np.sum(np.equal(val_list, v))
            return tree

    if len(val_cat) == 1:
        print("cat is zero")
        tree = {"root":None, "branches":{val_cat.pop(): len(val_list)}}
        return tree
    
    best_attr, val_cat = getBestSplitAttr(dataset, attrs)
        
    tree["root"] = best_attr
    attrs.pop(attrs.index(best_attr))    
    for v in val_cat:
        subd = splitDataset(dataset, best_attr, v)
        tree['branches'][v] = id3_build_tree(subd, copy.copy(attrs))
    return tree
 
#valid, if branches not in tree, return default
def tree_forward(data, tree, default=0):
    root_index = tree["root"]
    if root_index is None:
        minv = -1
        label = -1
        for k, v in tree['branches'].items():
            if minv < v:
                label = k
        return label
    
    v = data[root_index]
    branches = tree["branches"]
    if v not in branches:
        return default
    return tree_forward(data, branches[v], default=0)
    
if __name__ == '__main__':
    data = [
        [1, 1, 1]
        ,[1, 1, 1]
        ,[1, 0, 0]
        ,[0, 1, 0]
        ,[0, 1, 0]
    ]
    data = np.matrix(data)
    tree = id3_build_tree(data, [0, 1])
    print(tree)
    for d in data:
        print(tree_forward(d.tolist()[0], tree, 0))
    
