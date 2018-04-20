#!/usr/bin/python 
'''
decision tree base on ID3
'''
import numpy as np
import math

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
	new_ds = []
	for d in dataset:
		if d[attr_index] == value:
			t = d[:attr_index]
			t.extend(d[attr_index+1:])
			new_ds.append(t)
	return new_ds

def getBestSplitAttr(dataset):
    rols, cols = dataset.shape
    if cols == 1:
        return None, None
    features = cols- 1
    total_entrophy = cal_entrophy(dataset)
    max_gain = -1
    f_index = -1
    value_category = None
    for f in range(features):
        values = [v for v in dataset[f][f]]
        values = set(values)
        entrophy = 0
        for v in values:
            sd = splitDataset(dataset, f, v)
            entrophy -= cal_entrophy(sd)
        gain = total_entrophy - entrophy
        if gain > max_gain:
            f_index = f
            max_gain = gain
            value_category = values
    return f_index, value_category

def id3_build_tree(dataset):
    rols, cols = dataset.shape
    feature_num = cols - 1
    if feature_num == 0:
        return None
    tree = None
    if cols > 1:
        attr, vcat = getBestSplitAttr(dataset)
        tree = {attr:{}}
        for v in vcat:
            tree[attr][v] = id3_build_tree(splitDataset(dataset, attr, v))
    else:
        for v in datasetp[:, -1].T.tolist()[0]:
            tree[]
    return tree    

def data_map(dataset):
    rols, cols = dataset.shape
    
    vmap = []
    for col in range(cols):
        v_cat = dataset[:, col].T.tolist()[0]
        v_cat = list(set(v_cat))
        vmap.append(v_cat)
        for rol in range(rols): 
            dataset[rol, col] = v_cat.index(dataset[rol, col])
    return vmap, dataset.astype(int)
    
if __name__ == '__main__':
    data = [
        [1, 1, 'yes']
        ,[1, 1, 'yes']
        ,[1, 1, 'yes']
        ,[1, 0, 'no']
        ,[0, 1, 'no']
    ]
    data = np.matrix(data)
    vmap, data = data_map(data)
    print (vmap)
    print (data)
    print(id3_build_tree(data))	
	
