import os
import re
import numpy as np
import tree as Tree

'''
 1 : the patient should be fitted with hard contact lenses,
     2 : the patient should be fitted with soft contact lenses,
     3 : the patient should not be fitted with contact lenses.

    1. age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    2. spectacle prescription:  (1) myope, (2) hypermetrope
    3. astigmatic:     (1) no, (2) yes
    4. tear production rate:  (1) reduced, (2) normal

'''

def get_dataset(data_path):
    classes = ['hard', 'soft', 'no lenses']
    attrs = ['age', 'prescription', 'astigmatic', 'tear rate']
    attrv = [
                 ['young', 'pre-presbyopic', 'presbyopic']
                ,['myope', 'hypermetrope']
                ,['no', 'yes']
                ,['reduced', 'normal']
            ]
    data = []
    pattern = re.compile("\s+|\t+")
    with open(data_path, 'r') as fd :
        for line in fd.readlines():
            line = line.strip()
            line = pattern.sub(' ', line)
            info = line.split(' ')
            data.append(info[1:])
    return classes, attrs, attrv, np.matrix(data).astype(int) - 1

def anno_match(classes, attrs, attrv, tree):
    if "root" not in tree:
        #classify
        _tree = {}
        for k, v in tree.items():
            cls = classes[k]
            _tree[cls] = v
        return _tree
    _tree = {"root":None, "branches":{}}
    root_index = tree["root"]
    _tree["root"] = attrs[root_index]
    for v, t in tree['branches'].items():
        v = attrv[root_index][v]
        _tree["branches"][v] = anno_match(classes, attrs, attrv, t)
    return _tree

if __name__ == '__main__':
    classes, attrs, attrv, ds = get_dataset("/home/pi/playground/data/lenses/lenses.data")
    tree = Tree.id3_build_tree(ds, [x for x in range( len(attrs) )] )
    print(tree)
    print(anno_match(classes, attrs, attrv, tree))
