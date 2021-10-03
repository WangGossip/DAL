from operator import truediv
import numpy as np
from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn.functional as F
import time
from .strategy import Strategy
# *BMCore方法，贪婪求效用度-冗余度最大的样本，每次选bn再选n，这n个就用k-center-greedy的方法
class BMCore(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(BMCore, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    # -需要得到每个样本的完整计算概率；以及对应的潜变量；选取n个样本
    def query(self, n):
        # -一些常量
        beta = 5#这是候选样本集合要得到的数量
        bn = beta * n #候选集数量
        device = self.device

        # ~idxs_lb是一个bool数组，记录每一个样本点状态（是否被标记）
        idxs_state_labeled = self.idxs_lb.copy()
        lb_state_use = self.idxs_lb.copy()
        n_pool = self.n_pool
        n_lbd = np.sum(idxs_state_labeled)
        n_choose_use = n_lbd+bn#这是最终矩阵中应有的大小
        log_run = self.args.log_run
        T = self.args.timer
        T.start()
        time_start = time.time()

        delta = 0.1
        p = 1
        # size_fig = X[0].numel()
        # -其余数据处理
        # idxs_labeled = np.arange(n_pool)[idxs_lb]
        idxs_unlabeled = np.arange(n_pool)[~idxs_state_labeled]
        # probs, hide_z = self.predict_prob_bmal(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # ~需要所有的已标记以及挑选的部分未标记样本，迭代筛选；
        probs, hide_z = self.predict_prob_bmal(self.X, self.Y)
        probs_sorted, idxs = probs[idxs_unlabeled].sort(descending=True)# 计算效用度，只需要从unlabeled集合考虑即可
        #~排序后，是二维向量；第i个向量代表第i个样本的预测结果，均属于未标记池；结果是各个类别的预测概率
        # *先使用边缘采样
        # *先计算效用分数，max(δ,1-(c1-c2))
        V = probs_sorted[:, 0] - probs_sorted[:,1] #计算每个样本的效用度，此时U是一个一维数组，代表第i个样本的效用度
        V_score = torch.tensor([max(delta, 1-t) for t in V])
        idxs_candidate = V_score.sort(descending=True)[1][:bn] #~降序排列，这里记录的是~idx_lb集合中，选择的下标
        #~ 此时得到的候选集，长度为bn个，目的是挑选n个，元素含义是对应样本集的索引
        #- 需要用到的隐藏层有：标记集+候选集，记录下对应的下标 
        lb_state_use[idxs_unlabeled[idxs_candidate]] = True
        idxs_use = np.arange(n_pool)[lb_state_use]#在迭代过程中使用到的所有样本对应数据集中的下标

        lb_state_chooseed = np.zeros(n_choose_use, dtype=bool)
        # -给出使用集里的标记状态
        for i in range(n_choose_use):
            if idxs_state_labeled[idxs_use[i]]:
                lb_state_chooseed[i] = True

        # print('长度1为：{}；使用长度为：{}'.format(np.sum(lb_state_chooseed),n_choose_use))
        # 最终仅需要对hide_z进行计算，其长度应该为
        hide_z = hide_z[idxs_use]
        tmp_time = T.stop()
        log_run.logger.debug('计算效用分数部分结束，用时 {:.4f} s'.format(tmp_time))

        # *计算相似矩阵，使用余弦距离表示冗余，0.5M+0.5为冗余矩阵，范围[0,1]
        # 仅计算候选集合中的部分样本
        T.start()
        log_run.logger.debug('计算相似度开始')

        hide_z = hide_z.to(device)
        # hide_z = F.normalize(hide_z)

        dist_mat = hide_z.mm(hide_z.T)
        sq = dist_mat.diagonal().reshape(n_choose_use, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.T
        dist_mat = dist_mat.sqrt()

        tmp_time = T.stop()
        log_run.logger.debug('计算欧氏距离(l2-norm)部分结束，用时 {:.4f} s'.format(tmp_time))
        T.start()

        # -3. 选择要使用的矩阵，此时mat矩阵元素(i，j)代表了第i个未标记点距离第j个已标记点的距离
        # 需要的是候选集上各个点的状态

        # todo修改，只返回新加样本
        tmp_np_use = np.arange(n_choose_use)
        for i in range(n):
            mat_min = dist_mat[~lb_state_chooseed, :][:, lb_state_chooseed].min(axis=1)#~分别计算此时未标注集到标记集合的距离
            tmp_idx_ = mat_min.values.argmax()#~选择一个这样距离最大的样本，给出其对应纵坐标，这个下标是在未标记集中下标
            tmp_idx = tmp_np_use[~lb_state_chooseed][tmp_idx_]
            # print('id is {}, value is {}'.format(tmp_idx, lb_state_chooseed[~lb_state_chooseed][tmp_idx]))
            lb_state_chooseed[tmp_idx] = True

        # test
        print('长度为：{}；使用长度为：{}'.format(np.sum(lb_state_chooseed),n_choose_use))
        # test
        tmp_time = T.stop()
        log_run.logger.debug('贪婪筛选部分结束，用时 {:.4f} s'.format(tmp_time))


        time_use = time.time()-time_start
        log_run.logger.info('本次BMCore策略用时：{:.4f} s'.format(time_use))
        return idxs_use[lb_state_chooseed]
        # return idxs_unlabeled[idxs_candidate[idxs_state_choosed]]
        # return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]