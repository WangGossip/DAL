
# *存放各种自定义类，工具类
import csv
import time 
import os
import numpy as np
# *csv类，记录各种数据
# 文件名：数据
# init:参数args，文件名
# write_title：表头列表
# write_data： 一行内容
# close：关闭
class csv_results:
    def __init__(self, args, str_file='result.csv'):
        csv_path = os.path.join(args.out_path, str_file)
        csv_handler = open(csv_path, 'w', encoding = 'utf-8')
        csv_writer = csv.writer(csv_handler)
        self.csv_path = csv_path
        self.csv_handler = csv_handler
        self.csv_wirter = csv_writer
    
    # 构建表头
    def write_title(self, titles):
        self.csv_wirter.writerow(titles)
        self.count_cols = len(titles)

    # 添加一行内容
    def write_data(self, data):
        if len(data) == self.count_cols:
            self.csv_wirter.writerow(data)
    
    # 关闭表格
    def close(self):
        self.csv_handler.close()

# *一个时间类，作为一个计时器，可以在每次需要计时的时候记录当前用时，也可以返回累计时间、时间总和、平均时间等
class Timer:  #@save
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

# *计数类，用于统计每一次筛选之后各样本占比以及当次策略各样本占比
# ~功能：记录每一次采样后比例，以及总比例
# -需要参数：总共的采样次数；样本总数；类别总数
# *新功能：要能每次进行一个添加操作，实际上这个长度并不是固定的
class label_count:
    def __init__(self, sample_times, data_num, class_num=10) -> None:
        self.times = sample_times
        self.data_num = data_num
        self.class_num = class_num
        self.samples_count_each = [ [0 for col in range (class_num)] ]
        self.samples_count_sum = [ [0 for col in range (class_num)] ]
        self.samples_props_each = [ [0 for col in range (class_num)] ]
        self.samples_props_sum = [ [0 for col in range (class_num)] ]        
        # self.samples_count_each = [[0 for col in range (class_num)] for row in range(sample_times)]
        # self.samples_count_sum = [[0 for col in range (class_num)] for row in range(sample_times)]
        # self.samples_props_each = [[0 for col in range (class_num)] for row in range(sample_times)]
        # self.samples_props_sum = [[0 for col in range (class_num)] for row in range(sample_times)]
        pass

    # * 当新采样了一批样本之后，各种比例的记录，这里每次都增加，不考虑中途插入
    # 采样都是通过获取id实现，因此需要所有的标签数据
    def write_sampling_once(self, sample_idx, labels, sampling_time):
        class_num = self.class_num
        data_num = self.data_num
        count_each_tmp = [0] * class_num
        count_sum_tmp = [0] * class_num
        samples_props_each_tmp = [0] * class_num
        samples_props_sum_tmp = [0] * class_num
        # ~记录这一次的采样各个类别的数量、比例
        for id in sample_idx:
            count_each_tmp[labels[id]] +=1
            
        self.samples_count_each.append(count_each_tmp)

        for class_id in range(class_num):
            samples_props_each_tmp[class_id] = count_each_tmp[class_id] / data_num

        self.samples_props_each.append(samples_props_each_tmp)

        # ~记录此时已经采样的全部样本的数量、比例
        if sampling_time == 0:
            count_sum_tmp = count_each_tmp
        else:
            count_sum_tmp = [self.samples_count_sum[sampling_time-1][i]+count_each_tmp[i] for i in range(class_num)]

        self.samples_count_sum.append(count_sum_tmp)

        for class_id in range(class_num):
            samples_props_sum_tmp = count_sum_tmp[class_id] / data_num

        self.samples_props_sum.append(samples_props_sum_tmp)

    def get_count(self, sampling_time):
        return self.samples_props_each[sampling_time], self.samples_props_sum[sampling_time], self.samples_count_each[sampling_time], self.samples_count_sum[sampling_time]
