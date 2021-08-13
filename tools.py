
# *存放各种自定义类，工具类
import csv
import time 
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
# 需要参数：总共的采样次数；样本总数；
class label_count:
    def __init__(self, sample_times, num_labels) -> None:
        self.times = sample_times
        self.count = num_labels
        pass