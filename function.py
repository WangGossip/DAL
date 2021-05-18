import numpy as np
# *根据标记，统计样本比例
# idxs:一个数组，里面的元素对应着这一次选取的样本下标；labels：所有的标签
# 输入的labels是tensor向量，返回一个numpy数组
def get_mnist_prop(idxs, labels, len, count_class):
    prop_lbs = np.zeros(count_class)
    count_lbs = np.zeros(count_class)
    for i in range(0,len):
        count_lbs[labels[idxs[i]]] += 1
    for i in range(0,count_class):
        prop_lbs[i] = count_lbs[i]/len
    return prop_lbs