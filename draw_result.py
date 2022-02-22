import numpy as np
import matplotlib.pyplot as plt
import os


# *根据结果进行画图
# *需求：
# -输入：数据集名称，方法名称，使用次数，准确率数组
# -图像显示要求：均值曲线；最大值、最小值浅色区间；图例；横纵标题
# *一些常量设置
# methods_name = ['RS', 'BvsB', 'K-Center-Greedy', 'BALD', 'BMMC']

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
def draw_multi_data(list_data, outpath):
    # figname = dataset + 'acc.png'
    # savepath = os.path.join(outpath, figname)
    plt.savefig()
    return
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
def main():
    outpath='./acc_results_final_test'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    #~ MNIST数据
    list_mnist_data = use_mnist_data()
    # for i in range(6):
    #     data = list_mnist_data[i]
    for data in list_mnist_data:
        print('method:{}; 剩余训练集精度: {}\n平均精度:{}'.format(data.method, data.trpre_acc, data.list_finacc))
if __name__ == '__main__':
    main()