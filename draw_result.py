import numpy as np
import matplotlib.pyplot as plt
import os


# *根据结果进行画图
# *需求：
# -输入：数据集名称，方法名称，使用次数，准确率数组
# -图像显示要求：均值曲线；最大值、最小值浅色区间；图例；横纵标题
# *一些常量设置
methods_name = ['RS', 'BvsB', 'K-Center-Greedy', 'BALD', 'BMMC']

# ~设置一个类，数据集名称，方法名称，横坐标数组，纵坐标数组，全部数据，得到的平均精度；
class data_train_acc:
    def __init__(self, dataset, method, list_count):
        self.dataset = dataset
        self.method = method
        self.list_count = list_count
        self.list_acc = []
        self.list_finacc = []
        self.list_trpre_acc = []
    def add_accdata(self, acc):
        self.list_acc.append(acc)
    def add_trpre_acc(self, trpre_acc):
        self.list_trpre_acc.append(trpre_acc)
    # def add
def draw_multi_data(list_data, outpath):
    # figname = dataset + 'acc.png'
    # savepath = os.path.join(outpath, figname)
    plt.savefig()
    return
def use_mnist_data(list_mnist_data):
    
    return 
def main():
    outpath='./acc_results_final_test'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    #~ MNIST数据
    list_mnist_data = []
    use_mnist_data(list_mnist_data)
if __name__ == '__main__':
    main()