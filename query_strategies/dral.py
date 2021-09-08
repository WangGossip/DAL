from operator import truediv
import numpy as np
from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn.functional as F
import time
from .strategy import Strategy
# *DRAL方法，贪婪求效用度-冗余度最大的样本，每次选bn再选n
class DRAL(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(DRAL, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    # -需要得到每个样本的完整计算概率；以及对应的潜变量；选取n个样本
    def query(self, n):
        # -一些常量
        beta = 4#这是候选样本集合要得到的数量
        bn = beta * n #候选集数量
        idxs_choosed = []
        idxs_state_choosed = np.zeros(bn, dtype=bool)#~用于记录候选集合中选择状态

        # ~idxs_lb是一个bool数组，记录每一个样本点状态（是否被标记）
        idxs_state_labeled = self.idxs_lb
        n_pool = self.n_pool
        log_run = self.args.log_run
        T = self.args.timer
        T.start()
        time_start = time.time()

        delta = 0.1
        p = 1
        len_lbd = len(idxs_state_labeled)
        size_total = len_lbd + n #这是最终矩阵中应有的大小
        # size_fig = X[0].numel()
        # -其余数据处理
        # idxs_labeled = np.arange(n_pool)[idxs_lb]
        idxs_unlabeled = np.arange(n_pool)[~idxs_state_labeled]
        probs, hide_z = self.predict_prob_bmal(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)# 计算效用度，只需要从unlabeled集合考虑即可
        #~排序后，是二维向量；第i个向量代表第i个样本的预测结果，均属于未标记池；结果是各个类别的预测概率
        # *先使用边缘采样
        # *先计算效用分数，max(δ,1-(c1-c2))
        V = probs_sorted[:, 0] - probs_sorted[:,1] #计算每个样本的效用度，此时U是一个一维数组，代表第i个样本的效用度
        V_score = torch.tensor([max(delta, 1-t) for t in V])
        idxs_candidate = V_score.sort(descending=True)[1][:bn] #~降序排列，这里记录的是~idx_lb集合中，选择的下标
        #~ 此时得到的候选集，长度为bn个，目的是挑选n个，元素含义是对应样本集的索引
        hide_z = hide_z[idxs_candidate]
        tmp_time = T.stop()
        log_run.logger.debug('计算效用分数部分结束，用时 {:.4f} s'.format(tmp_time))

        # *计算相似矩阵，使用余弦距离表示冗余，0.5M+0.5为冗余矩阵，范围[0,1]
        # 仅计算候选集合中的部分样本
        T.start()
        # time_start2 = time.time()
        log_run.logger.debug('计算相似度开始')

        # Martix_sim = torch.zeros(len_X, len_X).to(self.device)
        hide_z = hide_z.to(self.device)
        hide_z = F.normalize(hide_z)
        Martix_sim = 0.5* hide_z.mm(hide_z.T) + 0.5

        tmp_time = T.stop()
        log_run.logger.debug('计算相似度部分结束，用时 {:.4f} s'.format(tmp_time))
        T.start()

        # -循环0：先计算最有代表性的样本
        id_first = torch.sum(Martix_sim, dim = 0).argmax() #距离越大，两者越像，说明越接近样本中心位置
        idxs_state_choosed[id_first] = True
        num_range = np.arange(bn)
        # -循环1-n，每次要寻找与当前集合最不相似的样本
        for i in range(1,n):
            # ~先计算所有未选择集中，对于当前已选择集合的距离
            idx_next_tmp = torch.sum(Martix_sim[idxs_state_choosed][:, ~idxs_state_choosed], dim = 0).argmin()
            idx_next = num_range[~idxs_state_choosed][idx_next_tmp]
            idxs_state_choosed[idx_next] = True

        tmp_time = T.stop()
        log_run.logger.debug('贪婪筛选部分结束，用时 {:.4f} s'.format(tmp_time))


        time_use = time.time()-time_start
        log_run.logger.info('本次BMAL策略用时：{:.4f} s'.format(time_use))
        return idxs_unlabeled[idxs_candidate[idxs_state_choosed]]