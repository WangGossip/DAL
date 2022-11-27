from logging import log
from numpy.ma.core import where
import torch
import torch.nn.functional as F
import time
import numpy as np
from .strategy import Strategy
# *最终版BMMC方法（不含dropout），贪婪求效用度-冗余度最大的样本，每次选bn再选n，b有多个取值；这n个就用最新版的k-center-greedy的方法
# *b个结果投票进行选择，对于票数低于1的再随机筛选
class BMMC_LC(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(BMMC_LC, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    # -需要得到每个样本的完整计算概率；以及对应的潜变量；选取n个样本
    def query(self, n):
        # 看当前运行了多少次/样本数目
        count = self.idxs_lb.sum()
        log_run = self.args.log_run
        if count < self.args.change_line:
            # -一些常量
            beta_min = self.args.beta_min
            beta_max = self.args.beta_max
            vote_line = self.args.vote_line #投票界限
            bn = beta_max * n #候选集数量
            device = self.device

            # ~idxs_lb是一个bool数组，记录每一个样本点状态（是否被标记）
            idxs_state_labeled = self.idxs_lb.copy()
            lb_state_use = self.idxs_lb.copy()
            n_pool = self.n_pool
            n_lbd = np.sum(idxs_state_labeled)#~已标注点的个数


            log_run = self.args.log_run
            T = self.args.timer
            T.start()
            time_start = time.time()

            delta = 0.1
            p = 1
            # size_fig = X[0].numel()
            # -其余数据处理
            idxs_labeled = np.arange(n_pool)[idxs_state_labeled]
            idxs_unlabeled = np.arange(n_pool)[~idxs_state_labeled]

            # *1. 借助不确定性进行排序，得到候选集 idxs_candidate（长度为bn）,代表未标记集的idx
            # ~需要所有的已标记以及挑选的部分未标记样本，迭代筛选；
            probs, hide_z = self.predict_prob_bmal(self.X, self.Y)
            hide_z = hide_z.numpy()#~ size为60000*1
            probs_sorted, idxs = probs[idxs_unlabeled].sort(descending=True)# 计算效用度，只需要从unlabeled集合考虑即可
            #~排序后，是二维向量；第i个向量代表第i个样本的预测结果，均属于未标记池；结果是各个类别的预测概率
            # *先使用边缘采样
            # *先计算效用分数，max(δ,1-(c1-c2))
            V = probs_sorted[:, 0] - probs_sorted[:,1] #计算每个样本的效用度，此时U是一个一维数组，代表第i个样本的效用度
            V_score = torch.tensor([max(delta, 1-t) for t in V])
            idxs_candidate = V_score.sort(descending=True)[1][:bn] #~降序排列，这里记录的是~idx_lb集合中，选择的下标
            #~ 此时得到的候选集，长度为bn个，目的是挑选n个，元素含义是未标记样本集中对应的索引
            #- 需要用到的隐藏层有：标记集+候选集，记录下对应的下标 
            lb_state_use[idxs_unlabeled[idxs_candidate]] = True
            idxs_use = np.arange(n_pool)[lb_state_use]#~在迭代过程中使用到的所有样本对应数据集中的下标

            # *需要一个投票数组，记录bn个候选集中，每一个分别得到的票数是多少
            # *id代表了对应的大候选集中下标，数值是得到的票数 
            vote_candidate = np.zeros(bn)
            vec_hide_labeled = hide_z[idxs_labeled]#~已标注集的隐向量

            tmp_time = T.stop()
            log_run.logger.debug('获取候选集部分结束，用时 {:.4f} s'.format(tmp_time))
            T.start()

            # *2. 多次循环，对于每次的候选集用coresets方法筛选
            for tmp_b in range(beta_min, beta_max+1):
                T.start()
                tmp_bn = tmp_b * n
                dists_between_sets = []
                idxs_candidate_tmpbn = idxs_candidate[:tmp_bn]
                # ~当前候选集中未标记池隐变量向量
                vec_hide_tmp_unlabeled = hide_z[idxs_unlabeled[idxs_candidate_tmpbn]]
                # *a.找到当前距离标记集最远的点
                for idx_unlabeled_i in idxs_candidate_tmpbn:
                    # ~获取第i（共tmp_bn）个点的隐向量
                    hide_unlbd_i = hide_z[idxs_unlabeled[idx_unlabeled_i]]
                    vec_hide_idxunlbd_i = np.array([hide_unlbd_i] * n_lbd)
                    # ~计算欧式距离
                    dist_vec_i = np.sum(np.square(vec_hide_idxunlbd_i - vec_hide_labeled), axis=1)
                    dist_unlbd_i = dist_vec_i.min()
                    # ~到已标记集的距离
                    dists_between_sets.append(dist_unlbd_i)
                dists_between_sets = np.array(dists_between_sets)
                #~这个idx是在idxs_candidate中的下标，记录在当前tmp_bn中的位置
                idx_last_candidate = np.argmax(dists_between_sets)
                idx_pool_last = idxs_unlabeled[idxs_candidate_tmpbn[idx_last_candidate]]
                vote_candidate[idx_last_candidate] += 1
                # *b. 循环n-1次，贪婪找最远点
                for count in range(1, n):
                    # -更新距离向量
                    hide_last_candidate = np.array([hide_z[idx_pool_last]]*tmp_bn)
                    dist_vec_tmp = np.sum(np.square(hide_last_candidate - vec_hide_tmp_unlabeled), axis=1)
                    # -拼接距离向量，取最小值作为新距离
                    dists_between_sets = np.min(np.concatenate((dist_vec_tmp.reshape(1,-1), dists_between_sets.reshape(1,-1)),axis=0),axis=0)
                    # *取新的最远点，加入投票，更新last_candidate
                    idx_last_candidate = np.argmax(dists_between_sets)
                    idx_pool_last = idxs_unlabeled[idxs_candidate_tmpbn[idx_last_candidate]]
                    vote_candidate[idx_last_candidate] += 1
                tmp_time = T.stop()
                log_run.logger.info('第{}次投票结束，此次投票用时{:.4f}s'.format(tmp_b-beta_min+1, tmp_time))

            # *3. 根据投票结果进行筛选
            # -先排序，看有多少票数高于1，从第几位开始是1
            state_vote_candidate = np.zeros(bn, dtype=bool)
            sorted_vote = np.argsort(vote_candidate)
            idxs_voted_multi = np.where(vote_candidate > vote_line)[0]
            len_voted_multi = len(idxs_voted_multi)
            if len_voted_multi >= n:
                # ~这是升序排列，选取最后的n个即可
                state_vote_candidate[sorted_vote[bn-n:]] = True
            else:
                idxs_voted_one = np.where(vote_candidate == vote_line)[0]
                len_voted_random = n - len_voted_multi
                # -票数为1的做随机选择
                idxs_tmp_one = np.random.choice(idxs_voted_one, len_voted_random, replace=False)
                state_vote_candidate[idxs_voted_multi] = True
                state_vote_candidate[idxs_tmp_one] = True

            log_run.logger.debug('票数大于二的个数为：{}'.format(len_voted_multi))
            time_use = time.time()-time_start
            log_run.logger.info('此次筛选用时：{:.4f}s'.format(time_use))
            result_idxs = idxs_unlabeled[idxs_candidate[state_vote_candidate]]
        else:
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:, 0] - probs_sorted[:,1]
            result_idxs = idxs_unlabeled[U.sort()[1][:n]]
        # 计算ld分数
        pred_all_state = self.predict_unlabeled[result_idxs]
        predict = pred_all_state.max(1)
        predict /= self.args.epochs
        score_all = predict.sum()
        log_run.logger.debug('LDAL离散度得分为：{}'.format(score_all))
        return result_idxs