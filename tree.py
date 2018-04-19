#!/usr/bin/python 
import numpy as np
import math

def cal_entrophy(dataset):
	values = {}
	for d in dataset:
		v = d[-1]
		if v not in values:
			values[v] = 0
		values[v] += 1
	entrophy = 0
	count = len(dataset)
	for v, c in values.items():
		pro = 1.0  * c / count
		entrophy -= pro * math.log(pro, 2)
	return entrophy

def splitDataset(dataset, attr_index, value):
	new_ds = []
	for d in dataset:
		if d[attr_index] == value:
			t = d[:attr_index]
			t.extend(d[attr_inedex+1:])
			new_ds.append(t)
	return new_ds

def chooseBestSplit(dataset):
	features = len(dataset[0]) - 1
	total_entrophy = cal_etrophy(dataset)
	max_gain = -1
	f_index = -1
	for f in range(features):
		values = [for v in dataset[f]]
		values = set(values)
		entrophy = 0
		for v in values:
			sd = splitDataset(dataset, f, v)
			entrophy -= cal_entrophy(sd)
		gain = total_entrophy - entrophy
		if gain > max_gain:
			f_index = f
			max_gain = gain
	return f_index

if __name__ == '__main__':
	data = [
		[1, 1, 'yes']
		,[1, 1, 'yes']
		,[1, 1, 'yes']
		,[1, 0, 'no']
		,[0, 1, 'no']
	]
	print(cal_entrophy(data))	
	
