from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import time
from .strategy import Strategy
# *Core-Set方法，使用K-Center-greedy，每次迭代贪婪找k个最远
# *思路：时间换空间，贪婪寻找，不怕计算量过大；余弦距离与欧氏距离成正比
class Core_Sets(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(Core_Sets, self).__init__(X, Y, idxs_lb, net, handler, args, device)
    def query(self, n):
        log_run = self.args.log_run
        T = self.args.timer
        n_pool = self.n_pool #样本总数
        T.start()
        time_start = time.time()

        lb_flag = self.idxs_lb.copy()
        # -1. 获取所有样本在模型下的隐藏层作为特征
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy() #~ size为60000*1
        
        tmp_time = T.stop()
        log_run.logger.debug('获取隐藏层特征部分结束，用时 {:.4f} s'.format(tmp_time))
        T.start()

        # -2. 迭代n次，每次贪婪寻找当前距离最远的点
        for count in range(n):
            T.start()
            dist_min = 0
            can_idx = 0
            # -得到当前的标记集和未标记集
            idxs_lbd_tmp = np.arange(n_pool)[lb_flag] #-当前的已标注样本集的对应下标
            idxs_unlbd_tmp = np.arange(n_pool)[~lb_flag]
            n_lbd_tmp = len(idxs_lbd_tmp)
            n_unlbd_tmp = len(idxs_unlbd_tmp)
            vec_lbd = embedding[idxs_lbd_tmp]

            for idx_unlbd_i in idxs_unlbd_tmp:
                # -a.计算一个未标记样本距离当前标记集的距离
                # ~算该点到所有点距离，最小的一个为其到集合的距离
                embed_i = embedding[idx_unlbd_i]
                vec_unlbd_i = np.array([embed_i]*n_lbd_tmp)
                # dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))欧式距离计算公式
                # torch.pairwise_distance
                dist_vec_i = np.sum(np.square(vec_lbd - vec_unlbd_i), axis=1)
                dist_idx_i = dist_vec_i.min()
                # -b.更新最远距离，添加当前idx到候选
                if dist_idx_i > dist_min:
                    dist_min = dist_idx_i
                    can_idx = idx_unlbd_i
            
            # -一次取多个，假设一共取100次
            # cau_times = 100
            # count_unlbd = 0
            # cau_once = int(n_unlbd_tmp/cau_times)
            # size_hide = embedding.shape[1]
            # vec_lbd = vec_lbd.reshape(1, n_lbd_tmp, size_hide)
            # for i in range(cau_times):
            #     idx_start = i*cau_once
            #     if i < cau_times-1 :
            #         cau_use_tmp = cau_once
            #     else:
            #         cau_use_tmp = n_unlbd_tmp - count_unlbd
            #     idx_end = idx_start + cau_use_tmp
            #     count_unlbd += cau_use_tmp
            #     # -取得这些距离，作为一个矩阵
            #     idxs_use_tmp = idxs_unlbd_tmp[idx_start:idx_end]
            #     embed_unlbd_tmp = embedding[idxs_use_tmp]#~这些是当次计算用到的隐变量

            #     # -计算距离                
            #     embed_unlbd_tmp = embed_unlbd_tmp.reshape(cau_use_tmp, 1, size_hide)#~方便计算，数组进行扩充
            #     vec_unlbd_i = np.repeat(embed_unlbd_tmp, n_lbd_tmp, axis=1)
            #     vec_lbd_i = np.repeat(vec_lbd, cau_use_tmp, axis=0)
            #     dist_vec_i = np.sum(np.square(vec_unlbd_i - vec_lbd_i), axis=2)
            #     dist_vec_i_fin = np.min(dist_vec_i, axis=1)
            #     #~这一批中选取最大值
            #     count_candidate = np.argmax(dist_vec_i_fin)
            #     dist_idx_i = np.max(dist_vec_i_fin)
            #     idx_unlbd_i = idxs_unlbd_tmp[count_candidate]
            #     # -更新最远距离，添加当前idx到候选
            #     if dist_idx_i > dist_min:
            #         dist_min = dist_idx_i
            #         can_idx = idx_unlbd_i
            # -迭代后得到一个最远值
            lb_flag[can_idx] = True
            tmp_time = T.stop()
            print('第{}次筛选结束，对应下标为:{}，用时 {:.2f} s'.format(count+1, can_idx, tmp_time))

        tmp_time = T.stop()
        time_use = time.time() - time_start
        log_run.logger.debug('贪婪得到K-Center覆盖部分结束，用时 {:.4f} s；本次采样总共用时{:.4f} s'.format(tmp_time, time_use))
        T.start()
        
        return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]