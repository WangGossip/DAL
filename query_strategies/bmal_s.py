import numpy as np
import torch
import time
from .strategy import Strategy
# *BMALS方法，贪婪求效用度-冗余度最大的样本，区别就是求冗余度式不考虑原始的样本
class BMALSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(BMALSampling, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    # -需要得到每个样本的完整计算概率；以及对应的潜变量；选取n个样本
    def query(self, n):
        # -一些常量
        X = self.X.to(self.device)#训练集数据，tensor
        # ~idxs_lb是一个bool数组，记录每一个样本点状态（是否被标记）
        idxs_lb = self.idxs_lb
        n_pool = self.n_pool
        log_run = self.args.log_run
        T = self.args.timer
        T.start()
        time_start = time.time()

        delta = 0.1
        p = 1
        len_lbd = len(idxs_lb)
        len_X = len(X)
        size_total = len_lbd + n #这是最终矩阵中应有的大小
        size_fig = X[0].numel()
        # -其余数据处理
        idxs_labeled = np.arange(n_pool)[idxs_lb]
        idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
        probs, hide_z = self.predict_prob_bmal(self.X, self.Y)
        probs_sorted, idxs = probs.sort(descending=True)
        #~排序后，是二维向量；第i个向量代表第i个样本的预测结果，均属于未标记池；结果是各个类别的预测概率
        # *先计算效用分数，max(δ,1-(c1-c2))
        V = probs_sorted[:, 0] - probs_sorted[:,1] #计算每个样本的效用度，此时U是一个一维数组，代表第i个样本的效用度
        V_score = torch.tensor([max(delta, 1-t) for t in V])

        tmp_time = T.stop()
        log_run.logger.debug('采样测试部分结束，用时 {:.4f} s'.format(tmp_time))

        # *计算冗余度
        # -计算所有样本之间的相似度，不管是否被标记过
        T.start()
        time_start2 = time.time()
        log_run.logger.debug('计算相似度行开始')
        Martix_sim = torch.zeros(len_X, len_X).to(self.device)
        hide_z = hide_z.to(self.device)

        for i in range(len_X):
            for j in range(i):
                # 计算两图片之间的冗余度，先转换成一维向量
                # Martix_sim[i][j] = torch.cosine_similarity(X[i].to(torch.float).reshape(size_fig), X[j].to(torch.float).reshape(size_fig), dim=0)
                Martix_sim[i][j] = torch.cosine_similarity(hide_z[i], hide_z[j], dim = 0)
        tmp_time = T.stop()
        log_run.logger.debug('计算相似度计算部分结束，用时 {:.4f} s'.format(tmp_time))

        for i in range(len_X):
            for j in range(i + 1, len_X):
                Martix_sim[i][j] = Martix_sim[j][i]
        time_use = time.time() - time_start2
        log_run.logger.debug('计算相似度结束，用时 {:.4f} s'.format(time_use))
        # -使用余弦相似度计算现有的冗余矩阵
        Martix_redundancy = torch.zeros(size_total, size_total)
        for i in range(len_lbd):
            for j in range(len_lbd):
                # i、j计数用，代表的是标记集合中的下标，还要在idxs_labeled中取真实下标
                Martix_redundancy[i][j]=p*V_score[idxs_labeled[i]]*Martix_sim[idxs_labeled[i]][idxs_labeled[j]]
        # -当前的分数U
        # -贪婪寻找最优的解法
        idxs_Q_max = []
        for _ in range(n):
            idxs_labeled = np.arange(n_pool)[idxs_lb]
            idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
            U_score = V_score[idxs_lb].sum()
            id_max_tmp = idxs_unlabeled[0]
            len_labeled = len(idxs_lb)
            Q_score_max = 0

            # -遍历未标记池
            for sample_id in idxs_unlabeled:
                M_tmp = Martix_redundancy
                # -计算此时的分数U
                U_score_tmp = U_score + V_score[sample_id]
                # -计算此时的冗余矩阵分数R
                # 先修改矩阵，此时对应的下标就是sample_id
                for i in range(len_labeled):
                    M_tmp[len_labeled][i] = p * V_score[sample_id] * Martix_sim[sample_id][idxs_labeled[i]]
                    M_tmp[i][len_labeled] = p * V_score[idxs_labeled[i]] * Martix_sim[idxs_labeled[i]][sample_id]
                R_score_tmp = M_tmp.norm()
                Q_score_tmp = U_score_tmp - R_score_tmp
                # -判断是否更新当前选择
                if Q_score_tmp > Q_score_max:
                    Q_score_max = Q_score_tmp
                    id_max_tmp = sample_id
            
            # -修改矩阵
            for i in range(len_labeled):
                Martix_redundancy[len_labeled][i] = p * V_score[id_max_tmp] * Martix_sim[id_max_tmp][idxs_labeled[i]]
                Martix_redundancy[i][len_labeled] = p * V_score[idxs_labeled[i]] * Martix_sim[idxs_labeled[i]][id_max_tmp]
            # -记录数据
            idxs_Q_max.append(id_max_tmp)
            idxs_lb[id_max_tmp] = True

        time_use = time.time()-time_start
        log_run.logger.info('本次BMAL策略用时：{:.4f} s'.format(time_use))
        return idxs_Q_max