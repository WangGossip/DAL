import numpy as np
import torch
from .strategy import Strategy
# *BMAL方法，贪婪求效用度-冗余度最大的样本
class BMAL(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(BMAL, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    # -需要得到每个样本的完整计算概率；以及对应的潜变量；选取n个样本
    def query(self, n):
        # -一些常量
        X = self.X#训练集数据，tensor
        # ~idxs_lb是一个bool数组，记录每一个样本点状态（是否被标记）
        idxs_lb = self.idxs_lb
        n_pool = self.n_pool
        delta = 0.1
        p = 1
        len_lbd = len(idxs_lb)
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
        # *计算冗余度
        # -计算所有样本之间的相似度
        Martix_sim = torch.zeros(size_total, size_total)
        for i in range(size_total):
            for j in range(size_total):
                if i == j :
                    Martix_sim[i][j] = torch.tensor(0)
                else:
                    # 计算两图片之间的冗余度，先转换成一维向量
                    Martix_sim[i][j] = torch.cosine_similarity(X[i].reshape(size_fig), X[j].reshape(size_fig), dim=0)

        # -使用余弦相似度计算现有的冗余矩阵
        Martix_redundancy = torch.zeros(size_total, size_total)
        for i in range(len_lbd):
            for j in range(len_lbd):
                # i、j计数用，代表的是标记集合中的下标，还要在idxs_labeled中取真实下标
                Martix_redundancy[i][j]=p*V_score[idxs_labeled[i]]*Martix_sim[idxs_labeled[i]][idxs_labeled[j]]
        # -当前的分数U
        U_score = V_score[idxs_lb].sum()
        U_score_tmp = U_score
        # -贪婪寻找最优的解法
        idxs_Q_max = []
        for id_candidate in range(n):
            Q_score_max = 0
            idxs_labeled = np.arange(n_pool)[idxs_lb]
            idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
            id_max_tmp = idxs_unlabeled[0]
            len_labeled = len(idxs_lb)
            # -遍历未标记池
            for sample_id in idxs_unlabeled:
                M_tmp = Martix_redundancy
                sample_tmp = X[sample_id]
                # -计算此时的分数U
                U_score_tmp = U_score + V_score[sample_id]
                # -计算此时的冗余矩阵分数R
                # 先修改矩阵，此时对应的下标就是sample_id
                for i in range(len_labeled):
                    M_tmp[len_labeled][i] = p * V_score[sample_id] * Martix_sim[sample_id][idxs_labeled[i]]
                    M_tmp[i][len_labeled] = p * V_score[idxs_labeled[i]] * Martix_sim[idxs_labeled[i]][sample_id]
                R_score = M_tmp.norm()
                Q_score_tmp = U_score_tmp - R_score
                # -判断是否更新当前选择
                if Q_score_tmp > Q_score_max:
                    Q_score_max = Q_score_tmp
                    U_score = U_score_tmp
                    id_max_tmp = id_candidate
            
            # -修改矩阵
            for i in range(len_labeled):
                Martix_redundancy[len_labeled][i] = p * V_score[id_max_tmp] * Martix_sim[id_max_tmp][idxs_labeled[i]]
                Martix_redundancy[i][len_labeled] = p * V_score[idxs_labeled[i]] * Martix_sim[idxs_labeled[i]][id_max_tmp]
            # -记录数据
            idxs_Q_max.append(id_max_tmp)
            idxs_lb[id_max_tmp] = True

        return idxs_Q_max