# encoding: utf-8
from cProfile import label
from turtle import color
from matplotlib import style
# from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import os


# *根据结果进行画图
# *需求：
# -输入：数据集名称，方法名称，使用次数，准确率数组
# -图像显示要求：均值曲线；最大值、最小值浅色区间；图例；横纵标题
# *一些常量设置
# methods_name = ['RS', 'BvsB', 'K-Center-Greedy', 'BALD', 'BMMC']
marker_type = ['.', 'o', '*', '<', '>', 'x', '1', '2', '3']
# ~设置一个类，数据集名称，方法名称，横坐标数组，纵坐标数组，全部数据，得到的平均精度；
class data_train_acc:
    def __init__(self, dataset, method, count_samples):
        self.dataset = dataset
        self.method = method
        self.count_samples = count_samples #~代表横坐标
        self.list_acc = []
        self.list_finacc = [] #~最终用于画图的精度列表
        self.list_trpre_acc = []
        self.trpre_acc = 0 #~在剩余训练集的精度
    def add_accdata(self, acc, trpre_acc):
        self.list_acc.append(acc)
        self.list_trpre_acc.append(trpre_acc)
    def add_final_acc(self, acc_mean, trpre_acc):
        self.list_finacc = acc_mean
        self.trpre_acc = trpre_acc
    def caculate_finacc(self):
        np_finacc = np.array(self.list_acc)
        self.list_finacc = np_finacc.mean(axis=0).tolist()
        self.trpre_acc = np.mean(self.list_trpre_acc)
    # def add

# *功能：根据一个data_train_acc列表进行画图
def draw_multi_data(list_data, outpath, str_exp, str_type='Acc'):
    DATANAME = list_data[0].dataset
    figname = DATANAME + '_' + str_exp + '_' + str_type + '.png'
    savepath = os.path.join(outpath, figname)
    count_samples_x = list_data[0].count_samples #~横坐标
    str_title = '{} {} Result'.format(DATANAME, str_type)
    count_gap = list_data[0].count_samples[1] - list_data[0].count_samples[0]
    count_total = list_data[0].count_samples[-1] + count_gap
    my_x_ticks = np.arange(0, count_total, count_gap)#原始数据有13个点，故此处为设置从0开始，间隔为1
    # my_y_ticks = np.arange(79, 98.5, 0.1)
    # 画图开始
    plt.figure(figsize=(8,8), dpi=120)
    # plt.rcParams['figure.figsize'] = (4.0, 8.0)
    # plt.rcParams['figure.dpi'] = 400
    # plt.rcParams['savefig.dpi'] = 400
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    id_data = 0
    for tmp_data in list_data:
        plt.plot(count_samples_x, [acc * 100 for acc in tmp_data.list_finacc], marker = marker_type[id_data], label = tmp_data.method)
        id_data += 1
        if tmp_data.method == 'BMMC':
            tmp_locx = count_samples_x[-3]
            tmp_x = count_samples_x[-1]
            tmp_y = tmp_data.list_finacc[-1] * 100 
            plt.text(tmp_locx, tmp_y, '({},{:.2f}%)'.format(tmp_x, tmp_y), bbox=dict(facecolor="dimgray", alpha=0))
            # *水平参考线
            plt.axhline(tmp_y, linestyle='--', color = 'grey')
            # plt.plot(count_samples_x, [tmp_y] * len(count_samples_x), color='g', linestyle='--')
    plt.title(str_title)
    plt.xlabel("Number of Labled Samples")
    plt.ylabel("Acc%")
    plt.legend(loc='lower right')
    plt.savefig(savepath)
    return

# jump0
def use_mnist_data_j0():
    list_mnist_data = []
    methods_name = ['LDAL', 'LDLC', 'RS']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_data.append(tmp_mnist_data)
    
    # *LDAL
    list_mnist_data[0].add_final_acc([0.80434, 0.87528, 0.9065, 0.92224, 0.93166, 0.93916, 0.94312, 0.94466, 0.9471, 0.95048], 0.9466)    
    # *LDLC
    list_mnist_data[1].add_final_acc([0.7958, 0.87332, 0.9053, 0.91648, 0.92746, 0.93902, 0.94188, 0.94312, 0.94834, 0.9499], 0.9459)    
    # *RS 
    list_mnist_data[2].add_final_acc([0.80092, 0.88232, 0.906, 0.92484, 0.93256, 0.9422, 0.94754, 0.95202, 0.9561, 0.95706], 0.9518)    
    # # *LC
    # list_mnist_data[3].add_accdata([0.789, 0.916, 0.9397, 0.9551, 0.9645, 0.9707, 0.975, 0.9761, 0.9788, 0.9795], 0.9814)
    # list_mnist_data[3].add_accdata([0.8313, 0.9022, 0.9339, 0.9532, 0.9659, 0.9706, 0.9746, 0.9759, 0.9785, 0.9783], 0.9820)
    # list_mnist_data[3].add_accdata([0.7709, 0.882, 0.9247, 0.9418, 0.961, 0.9641, 0.9712, 0.9752, 0.9782, 0.981], 0.9826)
    # list_mnist_data[3].add_accdata([0.7643, 0.886, 0.9304, 0.9525, 0.9635, 0.9663, 0.9711, 0.9769, 0.978, 0.9798], 0.9820)
    # list_mnist_data[3].add_accdata([0.8014, 0.9078, 0.9363, 0.9513, 0.9657, 0.9648, 0.9728, 0.9748, 0.9788, 0.9764], 0.9796)
    # list_mnist_data[3].caculate_finacc()

    return list_mnist_data

# jump20
def use_mnist_data_j10():
    list_mnist_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_data.append(tmp_mnist_data)
    
    # *LDAL
    list_mnist_data[0].add_final_acc([0.77488, 0.8616, 0.89346, 0.91596, 0.9288, 0.93992, 0.94418, 0.94514, 0.94836, 0.95172], 0.9474)    
    # *LDLC
    list_mnist_data[1].add_final_acc([0.81506, 0.87142, 0.90114, 0.91868, 0.93002, 0.93926, 0.94438, 0.94322, 0.9462, 0.95116], 0.9463)    
    # # *RS 
    # list_mnist_data[2].add_final_acc([0.80092, 0.88232, 0.906, 0.92484, 0.93256, 0.9422, 0.94754, 0.95202, 0.9561, 0.95706], 0.9518)    
    # # *LC
    # list_mnist_data[3].add_accdata([0.789, 0.916, 0.9397, 0.9551, 0.9645, 0.9707, 0.975, 0.9761, 0.9788, 0.9795], 0.9814)
    # list_mnist_data[3].add_accdata([0.8313, 0.9022, 0.9339, 0.9532, 0.9659, 0.9706, 0.9746, 0.9759, 0.9785, 0.9783], 0.9820)
    # list_mnist_data[3].add_accdata([0.7709, 0.882, 0.9247, 0.9418, 0.961, 0.9641, 0.9712, 0.9752, 0.9782, 0.981], 0.9826)
    # list_mnist_data[3].add_accdata([0.7643, 0.886, 0.9304, 0.9525, 0.9635, 0.9663, 0.9711, 0.9769, 0.978, 0.9798], 0.9820)
    # list_mnist_data[3].add_accdata([0.8014, 0.9078, 0.9363, 0.9513, 0.9657, 0.9648, 0.9728, 0.9748, 0.9788, 0.9764], 0.9796)
    # list_mnist_data[3].caculate_finacc()

    return list_mnist_data
# jump20
def use_mnist_data_j20():
    list_mnist_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_data.append(tmp_mnist_data)
    
    # *LDAL
    list_mnist_data[0].add_final_acc([0.79832, 0.87204, 0.9019, 0.91878, 0.9290, 0.93972, 0.9411, 0.94346, 0.9453, 0.9494], 0.9451)    
    # *LDLC
    list_mnist_data[1].add_final_acc([0.80356, 0.86842, 0.90106, 0.92356, 0.93084, 0.93802, 0.94382, 0.94454, 0.94672, 0.95104], 0.9465)    
    # # *RS 
    # list_mnist_data[2].add_final_acc([0.80092, 0.88232, 0.906, 0.92484, 0.93256, 0.9422, 0.94754, 0.95202, 0.9561, 0.95706], 0.9518)    
    # # *LC
    # list_mnist_data[3].add_accdata([0.789, 0.916, 0.9397, 0.9551, 0.9645, 0.9707, 0.975, 0.9761, 0.9788, 0.9795], 0.9814)
    # list_mnist_data[3].add_accdata([0.8313, 0.9022, 0.9339, 0.9532, 0.9659, 0.9706, 0.9746, 0.9759, 0.9785, 0.9783], 0.9820)
    # list_mnist_data[3].add_accdata([0.7709, 0.882, 0.9247, 0.9418, 0.961, 0.9641, 0.9712, 0.9752, 0.9782, 0.981], 0.9826)
    # list_mnist_data[3].add_accdata([0.7643, 0.886, 0.9304, 0.9525, 0.9635, 0.9663, 0.9711, 0.9769, 0.978, 0.9798], 0.9820)
    # list_mnist_data[3].add_accdata([0.8014, 0.9078, 0.9363, 0.9513, 0.9657, 0.9648, 0.9728, 0.9748, 0.9788, 0.9764], 0.9796)
    # list_mnist_data[3].caculate_finacc()

    return list_mnist_data

# jump20
def use_mnist_data_j30():
    list_mnist_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_data.append(tmp_mnist_data)
    
    # *LDAL
    list_mnist_data[0].add_final_acc([0.78898, 0.86556, 0.8981, 0.91608, 0.92842, 0.9337, 0.94346, 0.94216, 0.94648, 0.94822], 0.94398)    
    # *LDLC
    list_mnist_data[1].add_final_acc([0.79488, 0.8654, 0.89852, 0.91774, 0.9285, 0.93804, 0.94042, 0.94156, 0.94528, 0.9503], 0.9458)    
    # # *RS 
    # list_mnist_data[2].add_final_acc([0.80092, 0.88232, 0.906, 0.92484, 0.93256, 0.9422, 0.94754, 0.95202, 0.9561, 0.95706], 0.9518)    
    # # *LC
    # list_mnist_data[3].add_accdata([0.789, 0.916, 0.9397, 0.9551, 0.9645, 0.9707, 0.975, 0.9761, 0.9788, 0.9795], 0.9814)
    # list_mnist_data[3].add_accdata([0.8313, 0.9022, 0.9339, 0.9532, 0.9659, 0.9706, 0.9746, 0.9759, 0.9785, 0.9783], 0.9820)
    # list_mnist_data[3].add_accdata([0.7709, 0.882, 0.9247, 0.9418, 0.961, 0.9641, 0.9712, 0.9752, 0.9782, 0.981], 0.9826)
    # list_mnist_data[3].add_accdata([0.7643, 0.886, 0.9304, 0.9525, 0.9635, 0.9663, 0.9711, 0.9769, 0.978, 0.9798], 0.9820)
    # list_mnist_data[3].add_accdata([0.8014, 0.9078, 0.9363, 0.9513, 0.9657, 0.9648, 0.9728, 0.9748, 0.9788, 0.9764], 0.9796)
    # list_mnist_data[3].caculate_finacc()

    return list_mnist_data

# fm j10
def use_fashionmnist_data_j10():
    list_fashionmnist_data = []
    methods_name = ['LDAL', 'LDLC', 'RS']
    DATA_NAME = 'FashionMNIST'
    count_samples = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    for method_name in methods_name:
        tmp_fashionmnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_fashionmnist_data.append(tmp_fashionmnist_data)    
    # *RS 
    list_fashionmnist_data[2].add_final_acc([0.6192, 0.7073, 0.8025, 0.8466, 0.8655, 0.8752, 0.8834, 0.88718, 0.8891, 0.89112], 0.8981)
    # *LDLC
    list_fashionmnist_data[1].add_final_acc([0.62136, 0.74924, 0.81038, 0.84492, 0.86216, 0.8713, 0.87582, 0.8803, 0.88264, 0.88746], 0.8936)    
    # *LDAL
    list_fashionmnist_data[0].add_final_acc([0.61888, 0.7539, 0.81364, 0.84574, 0.8598, 0.86736, 0.87384, 0.87828, 0.88014, 0.88584], 0.8928)    
    return list_fashionmnist_data
# fm j40
def use_fashionmnist_data_j40():
    list_fashionmnist_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'FashionMNIST'
    count_samples = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    for method_name in methods_name:
        tmp_fashionmnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_fashionmnist_data.append(tmp_fashionmnist_data)    
    # *LDAL
    list_fashionmnist_data[0].add_final_acc([0.62176, 0.73518, 0.81282, 0.84334, 0.86068, 0.86882, 0.87566, 0.87996, 0.8818, 0.88592], 0.8938)    
    # *LDLC
    list_fashionmnist_data[1].add_final_acc([0.63634, 0.751, 0.81528, 0.84128, 0.86244, 0.8697, 0.8755, 0.88026, 0.8789, 0.88692], 0.8933)    
    return list_fashionmnist_data
# cf j10
def use_cifar10_data_j10():
    list_cifar10_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'CIFAR10'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10_data.append(tmp_cifar10_data)
    # *LDAL
    list_cifar10_data[0].add_final_acc([0.39802, 0.486634, 0.55374, 0.59516, 0.62454, 0.65334, 0.6732, 0.68986, 0.70708, 0.71758], 0.7240)
    # *LDLC
    list_cifar10_data[1].add_final_acc([0.41222, 0.50854, 0.57426, 0.61582, 0.646, 0.67242, 0.69582, 0.7105, 0.7253, 0.73952], 0.7430)
    return list_cifar10_data

# cf j60
def use_cifar10_data_j60():
    list_cifar10_data = []
    methods_name = ['LDAL', 'LDLC']
    DATA_NAME = 'CIFAR10'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10_data.append(tmp_cifar10_data)
    # *LDAL
    list_cifar10_data[0].add_final_acc([0.40436, 0.5072, 0.57026, 0.6119, 0.64254, 0.66874, 0.69116, 0.70774, 0.72562, 0.73466], 0.7386)
    # *LDLC
    list_cifar10_data[1].add_final_acc([0.40158, 0.50206, 0.5638, 0.6056, 0.63492, 0.66358, 0.68356, 0.6976, 0.71592, 0.73016], 0.7358)
    return list_cifar10_data


# *功能：处理MNIST实验1数据
def use_mnist_data():
    list_mnist_data = []
    methods_name = ['RS', 'BvSB', 'LC', 'K-Center-Greedy', 'BALD', 'BMMC']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_data.append(tmp_mnist_data)
    # *RS 
    list_mnist_data[0].add_final_acc([0.80092, 0.88232, 0.906, 0.92484, 0.93256, 0.9422, 0.94754, 0.95202, 0.9561, 0.95706], 0.9518)
    # *BvSB
    list_mnist_data[1].add_accdata([0.7638, 0.9105, 0.9393, 0.9545, 0.9644, 0.9694, 0.9742, 0.9765, 0.9789, 0.9798], 0.9814)
    list_mnist_data[1].add_accdata([0.818, 0.9016, 0.936, 0.9516, 0.9635, 0.9642, 0.9738, 0.9751, 0.977, 0.9798], 0.9805)
    list_mnist_data[1].add_accdata([0.8116, 0.9193, 0.945, 0.9562, 0.9636, 0.9734, 0.9749, 0.9778, 0.9789, 0.9799], 0.9820)
    list_mnist_data[1].add_accdata([0.7868, 0.8845, 0.9425, 0.9522, 0.9636, 0.9692, 0.9748, 0.9745, 0.9747, 0.9805], 0.9820)
    list_mnist_data[1].add_accdata([0.7794, 0.9127, 0.9442, 0.9566, 0.9661, 0.9744, 0.9754, 0.9764, 0.9803, 0.9781], 0.9810)
    list_mnist_data[1].caculate_finacc()
    # *LC
    list_mnist_data[2].add_accdata([0.789, 0.916, 0.9397, 0.9551, 0.9645, 0.9707, 0.975, 0.9761, 0.9788, 0.9795], 0.9814)
    list_mnist_data[2].add_accdata([0.8313, 0.9022, 0.9339, 0.9532, 0.9659, 0.9706, 0.9746, 0.9759, 0.9785, 0.9783], 0.9820)
    list_mnist_data[2].add_accdata([0.7709, 0.882, 0.9247, 0.9418, 0.961, 0.9641, 0.9712, 0.9752, 0.9782, 0.981], 0.9826)
    list_mnist_data[2].add_accdata([0.7643, 0.886, 0.9304, 0.9525, 0.9635, 0.9663, 0.9711, 0.9769, 0.978, 0.9798], 0.9820)
    list_mnist_data[2].add_accdata([0.8014, 0.9078, 0.9363, 0.9513, 0.9657, 0.9648, 0.9728, 0.9748, 0.9788, 0.9764], 0.9796)
    list_mnist_data[2].caculate_finacc()

    # *K-Center-Greedy
    list_mnist_data[3].add_final_acc([0.80638, 0.8917, 0.92572, 0.93706, 0.94158, 0.95512, 0.95982, 0.9619, 0.9664, 0.97072], 0.9702)
    # *BALD
    list_mnist_data[4].add_final_acc([0.79758, 0.88238, 0.92498, 0.94944, 0.96056, 0.96866, 0.97316, 0.97582, 0.97834, 0.98068], 0.9808)
    # *BMMC
    list_mnist_data[5].add_accdata([0.8229, 0.9228, 0.9377, 0.9531, 0.9563, 0.9683, 0.9704, 0.9743, 0.9779, 0.9806], 0.9813)
    list_mnist_data[5].add_accdata([0.8233, 0.9054, 0.9336, 0.9497, 0.9494, 0.9659, 0.9702, 0.9738, 0.981, 0.9827], 0.9827)
    list_mnist_data[5].add_accdata([0.7606, 0.8711, 0.9135, 0.9515, 0.9611, 0.9647, 0.9702, 0.9735, 0.9797, 0.9805], 0.9811)
    list_mnist_data[5].add_accdata([0.8295, 0.9219, 0.9464, 0.9582, 0.9599, 0.9703, 0.974, 0.9773, 0.9807, 0.9824], 0.9815)
    list_mnist_data[5].add_accdata([0.7931, 0.8939, 0.9234, 0.9537, 0.9601, 0.9692, 0.9727, 0.9787, 0.9794, 0.9814], 0.9827)
    list_mnist_data[5].caculate_finacc()
    return list_mnist_data

def use_fashionmnist_data():
    list_fashionmnist_data = []
    methods_name = ['RS', 'BvSB', 'LC', 'K-Center-Greedy', 'BALD', 'BMMC']
    DATA_NAME = 'FashionMNIST'
    count_samples = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    for method_name in methods_name:
        tmp_fashionmnist_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_fashionmnist_data.append(tmp_fashionmnist_data)
    # *RS 
    list_fashionmnist_data[0].add_final_acc([0.6192, 0.7073, 0.8025, 0.8466, 0.8655, 0.8752, 0.8834, 0.88718, 0.8891, 0.89112], 0.8981)
    # *BvSB
    list_fashionmnist_data[1].add_accdata([0.6651, 0.7621, 0.8416, 0.8736, 0.8855, 0.8942, 0.8976, 0.9031, 0.9053, 0.907], 0.9288)
    list_fashionmnist_data[1].add_accdata([0.7259, 0.8093, 0.866, 0.8773, 0.886, 0.8992, 0.9021, 0.909, 0.9072, 0.9085], 0.9268)
    list_fashionmnist_data[1].add_accdata([0.6536, 0.6189, 0.8325, 0.872, 0.8904, 0.8992, 0.9039, 0.9076, 0.9068, 0.9061], 0.9281)
    list_fashionmnist_data[1].add_accdata([0.6231, 0.7835, 0.8493, 0.8806, 0.8966, 0.9, 0.9057, 0.9099, 0.9098, 0.9118], 0.9260)
    list_fashionmnist_data[1].add_accdata([0.6441, 0.7485, 0.8314, 0.8654, 0.8838, 0.8947, 0.8959, 0.9025, 0.9018, 0.9082], 0.9279)
    list_fashionmnist_data[1].caculate_finacc()
    # *LC
    list_fashionmnist_data[2].add_accdata([0.649, 0.7793, 0.8114, 0.8611, 0.8902, 0.8984, 0.9029, 0.9059, 0.9088, 0.9097], 0.9282)
    list_fashionmnist_data[2].add_accdata([0.7039, 0.7836, 0.8085, 0.862, 0.8831, 0.8952, 0.9, 0.9051, 0.9066, 0.9094], 0.9291)
    list_fashionmnist_data[2].add_accdata([0.7187, 0.763, 0.822, 0.8558, 0.879, 0.8902, 0.9011, 0.9034, 0.9063, 0.91], 0.9267)
    list_fashionmnist_data[2].add_accdata([0.5613, 0.675, 0.7362, 0.7812, 0.8717, 0.8731, 0.893, 0.9001, 0.9041, 0.9071], 0.923)
    list_fashionmnist_data[2].add_accdata([0.6051, 0.6567, 0.7909, 0.8441, 0.878, 0.8876, 0.895, 0.9053, 0.907, 0.9083], 0.9286)
    list_fashionmnist_data[2].caculate_finacc()

    # *K-Center-Greedy
    list_fashionmnist_data[3].add_final_acc([0.56274, 0.63708, 0.70584, 0.80566, 0.84892, 0.87284, 0.88486, 0.89096, 0.89604, 0.9024], 0.9124)
    # *BALD
    list_fashionmnist_data[4].add_final_acc([0.6024, 0.6565, 0.70358, 0.77612, 0.82904, 0.85728, 0.8759, 0.88376, 0.88654, 0.89234], 0.8995)
    # *BMMC
    list_fashionmnist_data[5].add_accdata([0.6459, 0.7681, 0.8398, 0.8773, 0.8903, 0.9024, 0.9017, 0.9063, 0.9033, 0.9132], 0.9280)
    list_fashionmnist_data[5].add_accdata([0.6869, 0.7843, 0.8572, 0.8717, 0.8827, 0.8934, 0.8984, 0.9019, 0.9038, 0.9109], 0.9282)
    list_fashionmnist_data[5].add_accdata([0.6054, 0.6741, 0.8246, 0.8653, 0.8838, 0.8948, 0.9022, 0.9023, 0.9047, 0.9115], 0.9286)
    list_fashionmnist_data[5].add_accdata([0.5085, 0.687, 0.8041, 0.7797, 0.8824, 0.8957, 0.9024, 0.9074, 0.9089, 0.9127], 0.9273)
    list_fashionmnist_data[5].add_accdata([0.674, 0.7597, 0.8523, 0.8749, 0.8891, 0.8997, 0.9025, 0.9058, 0.9049, 0.9132], 0.9286)
    list_fashionmnist_data[5].caculate_finacc()    
    return list_fashionmnist_data

def use_cifar10_data():
    list_cifar10_data = []
    methods_name = ['RS', 'BvSB', 'LC', 'K-Center-Greedy', 'BALD', 'BMMC']
    DATA_NAME = 'CIFAR10'
    count_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for method_name in methods_name:
        tmp_cifar10_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10_data.append(tmp_cifar10_data)
    # *RS 
    list_cifar10_data[0].add_final_acc([0.5029, 0.6206, 0.68884, 0.73406, 0.76518, 0.7889, 0.80608, 0.8219, 0.83448, 0.84452], 0.8469)
    # *BvSB
    list_cifar10_data[1].add_final_acc([0.50022, 0.63576, 0.72092, 0.76973, 0.80175, 0.83083, 0.84807, 0.86129, 0.8732, 0.88204], 0.9215)
    # *LC
    list_cifar10_data[2].add_final_acc([0.499, 0.62908, 0.71801, 0.77023, 0.80569, 0.83288, 0.84975, 0.86268, 0.87544, 0.88283], 0.9212)
    # *K-Center-Greedy
    list_cifar10_data[3].add_final_acc([0.50832, 0.63629, 0.71363, 0.76144, 0.79183, 0.818487, 0.83776, 0.852007, 0.863473, 0.874], 0.9022)
    # *BALD
    list_cifar10_data[4].add_final_acc([0.50192, 0.63256, 0.71126, 0.76161, 0.79227, 0.82031, 0.83865, 0.85319, 0.86548, 0.87473], 0.9045)
    # *BMMC
    list_cifar10_data[5].add_accdata([0.4972, 0.647, 0.7301, 0.7821, 0.8048, 0.8327, 0.8513, 0.864, 0.8817, 0.8873], 0.9239)
    list_cifar10_data[5].add_accdata([0.5322, 0.6612, 0.745, 0.7847, 0.8144, 0.8352, 0.8533, 0.8665, 0.8765, 0.8857], 0.9224)
    list_cifar10_data[5].add_accdata([0.5111, 0.6505, 0.7336, 0.79, 0.8164, 0.8403, 0.8562, 0.8707, 0.8795, 0.8869], 0.9241)
    list_cifar10_data[5].add_accdata([0.5197, 0.664, 0.7386, 0.7759, 0.8096, 0.8302, 0.8525, 0.8657, 0.878, 0.8853], 0.9227)
    list_cifar10_data[5].add_accdata([0.5068, 0.6482, 0.7239, 0.7716, 0.8101, 0.8371, 0.8491, 0.8658, 0.8764, 0.8886], 0.9237)
    list_cifar10_data[5].caculate_finacc()        
    return list_cifar10_data

def use_cifar10im3_data():
    list_cifar10im3_data = []
    methods_name = ['BMMC', 'RS', 'BvSB', 'K-Center-Greedy']
    DATA_NAME = 'CIFAR10IM3'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10im3_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10im3_data.append(tmp_cifar10im3_data)
    # *BMMC
    list_cifar10im3_data[0].add_accdata([0.2882, 0.3738, 0.4273, 0.4611, 0.496, 0.5137, 0.5455, 0.5616, 0.5768, 0.5923], 0.9006)
    list_cifar10im3_data[0].add_accdata([0.2908, 0.3849, 0.445, 0.4725, 0.4937, 0.5233, 0.5443, 0.5701, 0.5815, 0.6014], 0.8989)
    list_cifar10im3_data[0].add_accdata([0.2941, 0.3945, 0.4359, 0.4854, 0.5109, 0.5424, 0.5623, 0.5703, 0.5867, 0.5974], 0.8927)
    list_cifar10im3_data[0].add_accdata([0.2941, 0.3549, 0.4154, 0.4675, 0.5011, 0.537, 0.551, 0.567, 0.5984, 0.6142], 0.9015)
    list_cifar10im3_data[0].add_accdata([0.2858, 0.362, 0.417, 0.469, 0.4913, 0.5136, 0.5458, 0.5555, 0.5659, 0.582], 0.8909)
    list_cifar10im3_data[0].caculate_finacc()
    # *RS
    list_cifar10im3_data[1].add_final_acc([0.3032, 0.338, 0.3819, 0.401, 0.4166, 0.4362, 0.4615, 0.469, 0.4859, 0.5147], 0.8311)
    # *BvSB
    list_cifar10im3_data[2].add_final_acc([0.2831, 0.3551, 0.3881, 0.4478, 0.4649, 0.4845, 0.5055, 0.5228, 0.5338, 0.5454], 0.8911)
    # *K-Center-Greedy
    list_cifar10im3_data[3].add_accdata([0.3029, 0.3666, 0.4247, 0.4661, 0.4878, 0.5172, 0.53, 0.5482, 0.5621, 0.5807], 0.8719)
    list_cifar10im3_data[3].add_accdata([0.2747, 0.3509, 0.4081, 0.4398, 0.4695, 0.4914, 0.5062, 0.5178, 0.5484, 0.5602], 0.8587)
    list_cifar10im3_data[3].add_accdata([0.2899, 0.3792, 0.4212, 0.4548, 0.4832, 0.4953, 0.5336, 0.5454, 0.5539, 0.5881], 0.8823)
    list_cifar10im3_data[3].add_accdata([0.297, 0.3511, 0.4195, 0.4677, 0.4877, 0.5069, 0.5272, 0.5389, 0.5696, 0.5966], 0.8852)
    list_cifar10im3_data[3].add_accdata([0.2974, 0.3533, 0.4033, 0.4563, 0.4836, 0.4992, 0.5203, 0.5357, 0.5471, 0.574], 0.8775)
    list_cifar10im3_data[3].caculate_finacc()

    return list_cifar10im3_data

def use_cifar10im2_data():
    list_cifar10im2_data = []
    methods_name = ['BMMC', 'RS', 'BvSB', 'K-Center-Greedy']
    DATA_NAME = 'CIFAR10IM2'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10im2_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10im2_data.append(tmp_cifar10im2_data)
    # *BMMC
    list_cifar10im2_data[0].add_accdata([0.3403, 0.4659, 0.5063, 0.5453, 0.5762, 0.612, 0.6468, 0.6633, 0.6924, 0.7126], 0.8554)
    list_cifar10im2_data[0].add_accdata([0.3254, 0.4419, 0.5203, 0.5675, 0.5927, 0.6155, 0.6503, 0.6678, 0.6914, 0.714], 0.8502)
    list_cifar10im2_data[0].add_accdata([0.3308, 0.4305, 0.5448, 0.577, 0.6052, 0.6339, 0.6528, 0.6826, 0.701, 0.7118], 0.8568)
    list_cifar10im2_data[0].add_accdata([0.3333, 0.4486, 0.5101, 0.5602, 0.5936, 0.6268, 0.647, 0.6642, 0.6832, 0.7089], 0.8542)
    list_cifar10im2_data[0].add_accdata([0.3373, 0.4415, 0.527, 0.5518, 0.5872, 0.6141, 0.6488, 0.6689, 0.686, 0.7214], 0.8584)
    list_cifar10im2_data[0].caculate_finacc()
    # *RS
    list_cifar10im2_data[1].add_final_acc([0.3297, 0.42174, 0.47544, 0.49978, 0.52614, 0.5509, 0.56928, 0.5825, 0.6028, 0.62348], 0.7901)
    # *BvSB
    list_cifar10im2_data[2].add_final_acc([0.32957, 0.42208, 0.4841, 0.52998, 0.5683, 0.59622, 0.62149, 0.646153, 0.667073, 0.688873], 0.8577)
    # *K-Center-Greedy
    list_cifar10im2_data[3].add_final_acc([0.33841, 0.4276, 0.492486, 0.53503, 0.57362, 0.603047, 0.627787, 0.650413, 0.668193, 0.690587], 0.8333)

    return list_cifar10im2_data

def use_cifar10im1_data():
    list_cifar10im1_data = []
    methods_name = ['BMMC', 'RS', 'BvSB', 'K-Center-Greedy']
    DATA_NAME = 'CIFAR10IM1'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10im1_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10im1_data.append(tmp_cifar10im1_data)    
    # *BMMC
    list_cifar10im1_data[0].add_accdata([0.3612, 0.4964, 0.5624, 0.6105, 0.6394, 0.6699, 0.6967, 0.7212, 0.736, 0.7452], 0.8284)
    list_cifar10im1_data[0].add_accdata([0.3487, 0.4715, 0.5358, 0.5922, 0.6271, 0.6662, 0.686, 0.7108, 0.7268, 0.7385], 0.8298)
    list_cifar10im1_data[0].add_accdata([0.3896, 0.4741, 0.5294, 0.5982, 0.64, 0.6649, 0.6896, 0.7094, 0.7333, 0.7361], 0.8293)
    list_cifar10im1_data[0].add_accdata([0.3536, 0.4908, 0.5478, 0.6038, 0.6425, 0.6688, 0.6958, 0.7081, 0.7144, 0.7467], 0.8234)
    list_cifar10im1_data[0].add_accdata([0.374, 0.4584, 0.5358, 0.5768, 0.611, 0.6461, 0.6726, 0.691, 0.716, 0.7365], 0.8179)
    list_cifar10im1_data[0].caculate_finacc()
    # *RS
    list_cifar10im1_data[1].add_final_acc([0.37482, 0.4562, 0.50398, 0.53932, 0.56924, 0.59904, 0.61846, 0.6385, 0.65676, 0.67598], 0.7695)
    # *BvSB
    list_cifar10im1_data[2].add_final_acc([0.36511, 0.465093, 0.53146, 0.57924, 0.612913, 0.641747, 0.6691, 0.689647, 0.708787, 0.72973], 0.8332)
    # *K-Center-Greedy
    list_cifar10im1_data[3].add_final_acc([0.36469, 0.47622, 0.53635, 0.58389, 0.61749, 0.64842, 0.67332, 0.69504, 0.71412, 0.73394], 0.8152)
    
    return list_cifar10im1_data
def use_cifar10im0_data():
    list_cifar10im0_data = []
    methods_name = ['BMMC', 'RS', 'BvSB', 'K-Center-Greedy']
    DATA_NAME = 'CIFAR10IM0'
    count_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for method_name in methods_name:
        tmp_cifar10im0_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10im0_data.append(tmp_cifar10im0_data)    
    # *BMMC
    list_cifar10im0_data[0].add_accdata([0.4164, 0.5265, 0.5973, 0.6476, 0.6776, 0.7092, 0.7371, 0.7562, 0.778, 0.7837], 0.8268)
    list_cifar10im0_data[0].add_accdata([0.4236, 0.522, 0.5954, 0.647, 0.6713, 0.702, 0.7321, 0.7475, 0.7659, 0.7788], 0.8160)
    list_cifar10im0_data[0].add_accdata([0.3975, 0.5057, 0.5952, 0.6388, 0.6777, 0.7023, 0.722, 0.7367, 0.7567, 0.7736], 0.8140)
    list_cifar10im0_data[0].add_accdata([0.4109, 0.5181, 0.5806, 0.6319, 0.6748, 0.701, 0.7346, 0.7547, 0.7687, 0.7918], 0.8281)
    list_cifar10im0_data[0].add_accdata([0.4012, 0.5164, 0.5746, 0.6189, 0.6608, 0.687, 0.7115, 0.7207, 0.7406, 0.7678], 0.8048)
    list_cifar10im0_data[0].caculate_finacc()
    # *RS
    list_cifar10im0_data[1].add_final_acc([0.37482, 0.4562, 0.50398, 0.53932, 0.56924, 0.59904, 0.61846, 0.6385, 0.65676, 0.67598], 0.7695)
    # *BvSB
    list_cifar10im0_data[2].add_final_acc([0.39377, 0.4941, 0.56462, 0.60446, 0.64427, 0.67215, 0.69305, 0.71839, 0.73699, 0.75138], 0.8033)
    # *K-Center-Greedy
    list_cifar10im0_data[3].add_final_acc([0.39268, 0.48979, 0.55744, 0.60281, 0.63576, 0.66323, 0.68442, 0.70803, 0.72781, 0.74133], 0.7811)
    
    return list_cifar10im0_data

def use_mnist_bm_data():
    list_mnist_bm_data = []
    methods_name = ['BMMC', 'BMCore4', 'BMCore5', 'BMCore6']
    DATA_NAME = 'MNIST'
    count_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for method_name in methods_name:
        tmp_mnist_bm_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_mnist_bm_data.append(tmp_mnist_bm_data)    
    # *BMMC
    list_mnist_bm_data[0].add_accdata([0.8229, 0.9228, 0.9377, 0.9531, 0.9563, 0.9683, 0.9704, 0.9743, 0.9779, 0.9806], 0.9813)
    list_mnist_bm_data[0].add_accdata([0.8233, 0.9054, 0.9336, 0.9497, 0.9494, 0.9659, 0.9702, 0.9738, 0.981, 0.9827], 0.9827)
    list_mnist_bm_data[0].add_accdata([0.7606, 0.8711, 0.9135, 0.9515, 0.9611, 0.9647, 0.9702, 0.9735, 0.9797, 0.9805], 0.9811)
    list_mnist_bm_data[0].add_accdata([0.8295, 0.9219, 0.9464, 0.9582, 0.9599, 0.9703, 0.974, 0.9773, 0.9807, 0.9824], 0.9815)
    list_mnist_bm_data[0].add_accdata([0.7931, 0.8939, 0.9234, 0.9537, 0.9601, 0.9692, 0.9727, 0.9787, 0.9794, 0.9814], 0.9827)
    list_mnist_bm_data[0].caculate_finacc()    
    # *BMCore4
    list_mnist_bm_data[1].add_accdata([0.8249, 0.9035, 0.9386, 0.9485, 0.9589, 0.9695, 0.9712, 0.9758, 0.9778, 0.9827], 0.9827)
    list_mnist_bm_data[1].add_accdata([0.8117, 0.9114, 0.9402, 0.9575, 0.9647, 0.9689, 0.9727, 0.9767, 0.9781, 0.9811], 0.9811)
    list_mnist_bm_data[1].add_accdata([0.8311, 0.9222, 0.9395, 0.9572, 0.9659, 0.9676, 0.9713, 0.9725, 0.9737, 0.9785], 0.9796)
    list_mnist_bm_data[1].add_accdata([0.8006, 0.9057, 0.9493, 0.9631, 0.9657, 0.9728, 0.9747, 0.9777, 0.9782, 0.9805], 0.9823)
    list_mnist_bm_data[1].add_accdata([0.7823, 0.9024, 0.9429, 0.956, 0.9636, 0.9704, 0.974, 0.9769, 0.9797, 0.9797], 0.9808)
    list_mnist_bm_data[1].caculate_finacc()
    # *BMCore5
    list_mnist_bm_data[2].add_final_acc([0.80052, 0.90478, 0.93459, 0.95113, 0.96069, 0.96698, 0.97173, 0.97394, 0.97661, 0.9785], 0.9804)
    # *BMCore6
    list_mnist_bm_data[3].add_final_acc([0.79646, 0.90578, 0.93544, 0.95339, 0.96116, 0.96842, 0.9726, 0.97437, 0.97704, 0.97853], 0.9807)

    return list_mnist_bm_data

def use_fashionmnist_bm_data():
    list_fashionmnist_bm_data = []
    methods_name = ['BMMC', 'BMCore4', 'BMCore5', 'BMCore6']
    DATA_NAME = 'FashionMNIST'
    count_samples = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    for method_name in methods_name:
        tmp_fashionmnist_bm_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_fashionmnist_bm_data.append(tmp_fashionmnist_bm_data)    
    # *BMMC
    list_fashionmnist_bm_data[0].add_accdata([0.6459, 0.7681, 0.8398, 0.8773, 0.8903, 0.9024, 0.9017, 0.9063, 0.9033, 0.9132], 0.9280)
    list_fashionmnist_bm_data[0].add_accdata([0.6869, 0.7843, 0.8572, 0.8717, 0.8827, 0.8934, 0.8984, 0.9019, 0.9038, 0.9109], 0.9282)
    list_fashionmnist_bm_data[0].add_accdata([0.6054, 0.6741, 0.8246, 0.8653, 0.8838, 0.8948, 0.9022, 0.9023, 0.9047, 0.9115], 0.9286)
    list_fashionmnist_bm_data[0].add_accdata([0.5085, 0.687, 0.8041, 0.7797, 0.8824, 0.8957, 0.9024, 0.9074, 0.9089, 0.9127], 0.9273)
    list_fashionmnist_bm_data[0].add_accdata([0.674, 0.7597, 0.8523, 0.8749, 0.8891, 0.8997, 0.9025, 0.9058, 0.9049, 0.9132], 0.9286)    
    
    list_fashionmnist_bm_data[0].caculate_finacc()        
    # *BMCore4
    list_fashionmnist_bm_data[1].add_final_acc([0.60523, 0.68833, 0.78222, 0.84648, 0.88127, 0.89294, 0.8989, 0.90392, 0.90369, 0.90963], 0.92786)
    # *BMCore5
    list_fashionmnist_bm_data[2].add_final_acc([0.61247, 0.71989, 0.81862, 0.86243, 0.88477, 0.894, 0.89998, 0.90443, 0.90515, 0.91014], 0.92784)
    # *BMCore6
    list_fashionmnist_bm_data[3].add_final_acc([0.59952, 0.67126, 0.7672, 0.81803, 0.85588, 0.87902, 0.89567, 0.90146, 0.9035, 0.90919], 0.9275)
    return list_fashionmnist_bm_data

def use_cifar10_bm_data():
    list_cifar10_bm_data = []
    methods_name = ['BMMC', 'BMCore4', 'BMCore5', 'BMCore6']
    DATA_NAME = 'CIFAR10'
    count_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for method_name in methods_name:
        tmp_cifar10_bm_data = data_train_acc(DATA_NAME, method_name, count_samples)
        list_cifar10_bm_data.append(tmp_cifar10_bm_data)
    # *BMMC
    list_cifar10_bm_data[0].add_accdata([0.4972, 0.647, 0.7301, 0.7821, 0.8048, 0.8327, 0.8513, 0.864, 0.8817, 0.8873], 0.92165)
    list_cifar10_bm_data[0].add_accdata([0.5322, 0.6612, 0.745, 0.7847, 0.8144, 0.8352, 0.8533, 0.8665, 0.8765, 0.8857], 0.9224)
    list_cifar10_bm_data[0].add_accdata([0.5198, 0.6653, 0.7468, 0.787, 0.812, 0.8339, 0.8524, 0.8654, 0.8782, 0.8852], 0.9201)
    list_cifar10_bm_data[0].add_accdata([0.5197, 0.664, 0.7386, 0.7759, 0.8096, 0.8302, 0.8525, 0.8657, 0.878, 0.8853], 0.9227)
    list_cifar10_bm_data[0].add_accdata([0.4999, 0.6485, 0.7379, 0.7826, 0.8148, 0.844, 0.8542, 0.8724, 0.8816, 0.8852], 0.9224)
    list_cifar10_bm_data[0].caculate_finacc()            
    # *BMCore4
    list_cifar10_bm_data[1].add_final_acc([0.5071, 0.64706, 0.73113, 0.77673, 0.80756, 0.83346, 0.85081, 0.86557, 0.87548, 0.8850], 0.9227)
    # *BMCore5
    list_cifar10_bm_data[2].add_final_acc([0.50679, 0.64538, 0.72568, 0.7779, 0.80755, 0.83256, 0.84942, 0.86333, 0.87525, 0.8834], 0.9214)
    # *BMCore6
    list_cifar10_bm_data[3].add_final_acc([0.50461, 0.64562, 0.72689, 0.77665, 0.80462, 0.83138, 0.84993, 0.8622, 0.8746, 0.88296], 0.9202)
    return list_cifar10_bm_data

def main():
    # outpath='./acc_results_final'
    outpath = './test_results_424'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    # *EXP1部分
    # todo CIFAR10-BMMC部分数据待修改
    # #~ MNIST数据
    # list_mnist_data0 = use_mnist_data_j0()
    # draw_multi_data(list_mnist_data0, outpath, 'EXP0')
    
    # #~ MNIST数据
    # list_mnist_data10 = use_mnist_data_j10()
    # draw_multi_data(list_mnist_data10, outpath, 'EXP10')

    # #~ MNIST数据
    # list_mnist_data20 = use_mnist_data_j20()
    # draw_multi_data(list_mnist_data20, outpath, 'EXP20')

    # #~ MNIST数据
    # list_mnist_data30 = use_mnist_data_j30()
    # draw_multi_data(list_mnist_data30, outpath, 'EXP30')

    # for data in list_mnist_data0:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # for data in list_mnist_data10:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))  
    # for data in list_mnist_data20:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))        
    # for data in list_mnist_data30:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))        
    # ~ 0619数据
    list_fashionmnist_data_j10 = use_fashionmnist_data_j10()
    draw_multi_data(list_fashionmnist_data_j10, outpath, 'EXPFM10')
    list_fashionmnist_data_j40 = use_fashionmnist_data_j40()
    draw_multi_data(list_fashionmnist_data_j40, outpath, 'EXPFM40')
    list_cifar10_data_j10 = use_cifar10_data_j10()
    draw_multi_data(list_cifar10_data_j10, outpath, 'EXPCF10')
    list_cifar10_data_j60 = use_cifar10_data_j60()
    draw_multi_data(list_cifar10_data_j60, outpath, 'EXPCF60')

    for data in list_fashionmnist_data_j10:
        print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    for data in list_fashionmnist_data_j40:
        print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    for data in list_cifar10_data_j10:
        print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    for data in list_cifar10_data_j60:
        print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))

    # # *EXP2部分，不平衡数据集
    # list_cifar10im3_data = use_cifar10im3_data()
    # draw_multi_data(list_cifar10im3_data, outpath, 'EXP2')
    # for data in list_cifar10im3_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # list_cifar10im2_data = use_cifar10im2_data()
    # draw_multi_data(list_cifar10im2_data, outpath, 'EXP2')
    # for data in list_cifar10im2_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # list_cifar10im1_data = use_cifar10im1_data()
    # draw_multi_data(list_cifar10im1_data, outpath, 'EXP2')
    # for data in list_cifar10im1_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # list_cifar10im0_data = use_cifar10im0_data()
    # draw_multi_data(list_cifar10im0_data, outpath, 'EXP2')
    # for data in list_cifar10im0_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))

    # # *EXP3部分，对比一次采样和多次投票
    # #~ MNIST数据
    # list_mnist_bm_data = use_mnist_bm_data()
    # draw_multi_data(list_mnist_bm_data, outpath, 'EXP3')
    # # ~FashionMNIST数据
    # list_fashionmnist_bm_data = use_fashionmnist_bm_data()
    # draw_multi_data(list_fashionmnist_bm_data, outpath, 'EXP3')
    # # ~CIFAR10数据
    # list_cifar10_bm_data = use_cifar10_bm_data()
    # draw_multi_data(list_cifar10_bm_data, outpath, 'EXP3')
    # for data in list_mnist_bm_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # for data in list_fashionmnist_bm_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
    # for data in list_cifar10_bm_data:
    #     print('data:{}; method:{}; 剩余训练集精度: {}; 测试集精度为：{}'.format(data.dataset, data.method, data.trpre_acc, data.list_finacc[-1]))
if __name__ == '__main__':
    main()