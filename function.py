import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt 
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


# *路径相关函数
# 修改 out-path，这是用来存储最终的log、csv以及对应的图片的
def get_results_dir(args):
    # args.
    time_start = time.localtime()
    path_results_tmp = '{}-{}-{}'.format(args.model_name, args.dataset, time.strftime(".%Y-%m-%d-%H:%M:%S", time_start))
    path_results_fin = os.path.join(args.out_path, path_results_tmp)
    if not os.path.exists(path_results_fin):
        os.makedirs(path_results_fin)
    args.out_path = path_results_fin
    return

# *画图相关函数
# 画loss（训练）的图
# todo 思考这个图怎么画出来
# ~实际用到的参数：全部loss数值；epochs的数量（args）；采样频率
# -横坐标：epoch；纵坐标loss
# 需要包括：loss变化趋势（纵坐标），横坐标是epoch变化（需要一个虚线），实际横坐标是epoch.iter, 
# args中应包含保存的路径
# csv格式：[epoch，iter，loss]
def draw_trloss(args, sample_times=5, fig_name='tr_loss.png'):
    # 其他参数
    n_epoch = args.epochs
    save_path = os.path.join(args.out_path, fig_name)
    csv_path = args.csv_record_trloss.csv_path
    # test
    # 手动赋值：save_path、csv_path
    # 获取数据
    f = open(csv_path, 'r')
    csv_reader = csv.reader(f)
    headline = next(csv_reader)
    count_data = 0
    iter_idxs = []
    loss = []
    #~ 读取的str格式，应转换为对应的int或float
    for row in csv_reader:
        count_data += 1             #计数，有多少行 
        iter_idxs.append(int(row[1]))
        loss.append(float(row[2]))
    f.close()
    # -设置横坐标，尺度为1
    # 设置采样相关参数，即每一个迭代epoch中采样几次
    # sample_times = 10
    # epoch，虚线分界；以及采样后的数据
    print('count_data is {}'.format(count_data))
    space_epoch = int(count_data/n_epoch)
    space_sample = int(space_epoch/sample_times)
    print('space_epoch is {}, space_sample is {}'.format(space_epoch, space_sample))
    # epoch_lines = []
    sample_loss_x = []
    sample_loss_y = []

    for i in range(sample_times*n_epoch):
        tmp_x = i*space_sample
        sample_loss_x.append(tmp_x)
        sample_loss_y.append(loss[tmp_x])
    max_loss_y = max(sample_loss_y)
    min_loss_y = min(sample_loss_y)
    # 开始画图
    plt.figure()
    plt.plot(sample_loss_x, sample_loss_y)
    # 添加虚线绘图
    for i in range(1, n_epoch):
        plt.vlines(i*space_epoch, min_loss_y, max_loss_y, colors='g', linestyles='--')
        # epoch_lines.append(i*space_epoch)    
    plt.title("LossResult")
    plt.xlabel("train")
    plt.ylabel("loss")

    plt.savefig(save_path)

# *画acc结果
# -横坐标，采样的样本数目；纵坐标，loss以及acc
def draw_tracc(args, fig_name='tr_acc.png'):
    # 参数处理
    n_epoch = args.epochs
    save_path = os.path.join(args.out_path, fig_name)
    csv_path = args.csv_record_tracc.csv_path
    # 读取数据
    f = open(csv_path, 'r')
    csv_reader = csv.reader(f)
    headline = next(csv_reader)
    count_sample = []
    te_acc = []
    te_loss = []
    for row in csv_reader:
        count_sample.append(int(row[1]))
        te_loss.append(float(row[3]))
        te_acc.append(float(row[2]))
    f.close()
    # 画图部分
    fig = plt.figure()
    ax_loss = fig.add_subplot(111)
    ax_loss.plot(count_sample, te_loss, 'r', label='test_loss')
    ax_loss.legend(loc=1)
    ax_loss.set_ylabel('Loss for each epoch')
    ax_acc = ax_loss.twinx() #~重点利用这个函数
    ax_acc.plot(count_sample, te_acc, 'g', label='test_acc')
    ax_acc.legend(loc=2)
    ax_acc.set_ylabel('Acc for after epoch')
    ax_acc.set_xlabel('Epoch')
    plt.title("Loss&ACC Result")
    plt.savefig(save_path)

# todo 展示每一次变化的比例
# *样本比例条形图
# -横坐标:采样次数；纵坐标：样本比例数目、不同样本对应的标签
# -需要的数据：颜色列表、样本标签名称、所有的数据
# ~plt.bar需要的参数包括bottom，是一个累加的过程
def draw_samples_prop(args, data_count, labels_name, fig_name):
    # 参数处理
    save_path = os.path.join(args.out_path, fig_name)
    colors_list = ['red','orange','yellow','green','cyan','blue','purple','pink','magenta','brown']
    sample_times = args.times + 1   #采样次数，从0开始直到最高
    # 获取数据
    # 需要的数据形式：每一类是一个列表，先类后次；实际的形式：先按次数排列；因此做一次转置
    data_use = np.array(data_count).T
    # x_time_fig = np.arange(sample_times + 1)
    # labels_n = len(labels_name)
    tmp_bottom = np.zeros(sample_times)
    # 画图
    plt.figure()
    bar_width=0.2
    # print(len(data_use), len(labels_name))
    for list, color, label in zip(data_use, colors_list, labels_name):
        plt.bar(np.arange(sample_times),
                list,
                color = color,
                width=bar_width,
                bottom=tmp_bottom,
                label=label
        )
        tmp_bottom = np.array([i+j for i,j in zip(tmp_bottom, list)])

    plt.xticks(np.arange(sample_times))#定义柱子名称
    plt.legend(loc=2)#图例在左边
    plt.title(fig_name[:-4])
    plt.savefig(save_path)