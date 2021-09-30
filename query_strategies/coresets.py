import numpy as np
import torch
import time
from .strategy import Strategy
# *Core-Set方法，使用K-Center-greedy，每次迭代贪婪找k个最远
# !做不了，运算量还是太大
class CoreSets(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(CoreSets, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    def query(self, n):
        log_run = self.args.log_run
        T = self.args.timer
        T.start()
        time_start = time.time()

        lb_flag = self.idxs_lb.copy()
        # -1. 获取所有样本在模型下的隐藏层作为特征
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy() #~ size为60000*1
        
        tmp_time = T.stop()
        log_run.logger.debug('获取隐藏层特征部分结束，用时 {:.4f} s'.format(tmp_time))
        T.start()

        # -2. 计算矩阵的l2-norm，即计算每两个样本之间的欧氏距离
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        tmp_time = T.stop()
        log_run.logger.debug('计算矩阵间欧式距离部分结束，用时 {:.4f} s'.format(tmp_time))
        T.start()

        # -3. 选择要使用的矩阵，此时mat矩阵元素(i，j)代表了第i个未标记点距离第j个已标记点的距离
        mat = dist_mat[~lb_flag, :][:, lb_flag]

        for i in range(n):
            mat_min = mat.min(axis=1)#~分别计算此时未标注集到标记集合的距离
            q_idx_ = mat_min.argmax()#~选择一个这样距离最大的样本
            q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
            #~ 把新样本添加到mat矩阵
            lb_flag[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        tmp_time = T.stop()
        time_use = T.time() - time_start
        log_run.logger.debug('贪婪得到K-Center覆盖部分结束，用时 {:.4f} s；本次采样总共用时{:.4f} s'.format(tmp_time, time_use))
        T.start()
        
        return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]