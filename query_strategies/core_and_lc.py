from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import time
from .strategy import Strategy

# todo 添加ldal作为评价标准；
# todo 计算ldal的值;
# * 一种两阶段的方法，先用代表性，后用不确定性
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import time
from .strategy import Strategy

class CoreLC(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(CoreLC, self).__init__(X, Y, idxs_lb, net, handler, args, device)
    def query(self, n):
        # 看当前运行了多少次/样本数目
        count = self.idxs_lb.sum()
        log_run = self.args.log_run
        if count < self.args.change_line:
            T = self.args.timer
            n_pool = self.n_pool #样本总数
            T.start()
            time_start = time.time()

            lb_flag = self.idxs_lb.copy()
            # *1. 获取所有样本在模型下的隐藏层作为特征
            embedding = self.get_embedding(self.X, self.Y)
            embedding = embedding.numpy() #~ size为60000*1
            
            tmp_time = T.stop()
            log_run.logger.debug('获取隐藏层特征部分结束，用时 {:.4f} s'.format(tmp_time))
            T.start()

            # *2. 先找到距离最远的一个点
            dists_between_data = []
            # -得到当前的标记集和未标记集
            idxs_lbd_tmp = np.arange(n_pool)[lb_flag] #-当前的已标注样本集的对应下标
            idxs_unlbd_tmp = np.arange(n_pool)[~lb_flag]
            n_lbd_tmp = len(idxs_lbd_tmp)
            n_unlbd_tmp = len(idxs_unlbd_tmp)
            vec_lbd = embedding[idxs_lbd_tmp]
            vec_unlbd = embedding[idxs_unlbd_tmp]
            for vec_i in vec_unlbd:
                # -a.计算一个未标记样本距离当前标记集的距离
                # ~算该点到所有点距离，最小的一个为其到集合的距离
                vec_unlbd_i = np.array([vec_i]*n_lbd_tmp)
                # dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))欧式距离计算公式
                dist_vec_i = np.sum(np.square(vec_lbd - vec_unlbd_i), axis=1)#不做开方，减小运算量
                dist_idx_i = dist_vec_i.min()        
                dists_between_data.append(dist_idx_i)

            dists_between_data = np.array(dists_between_data)
            idx_last_candidate = np.argmax(dists_between_data)
            # dists_between_data[idx_last_candidate] = 0
            lb_flag[idxs_unlbd_tmp[idx_last_candidate]]=True

            tmp_time = T.stop()
            log_run.logger.debug('获取第一个最远点，用时 {:.4f} s'.format(tmp_time))


            T.start()
            # *3. 循环n-1次，找到其余的最小距离
            for count in range(1,n):
                # *计算新的距离，更新当前最小距离
                embed_last_candidate = np.array([vec_unlbd[idx_last_candidate]] * n_unlbd_tmp)
                dist_vec_tmp = np.sum(np.square(embed_last_candidate - vec_unlbd), axis=1)
                # -拼接并取最小值，得到新的距离数组
                dists_between_data = np.min(np.concatenate((dist_vec_tmp.reshape(1,-1), dists_between_data.reshape(1,-1)),axis=0), axis=0)
                idx_last_candidate = np.argmax(dists_between_data)#~取新的最远点
                # dists_between_data[idx_last_candidate] = 0
                lb_flag[idxs_unlbd_tmp[idx_last_candidate]]=True
                # print('第{}个样本获取结束,临时idx为：{}'.format(count,idx_last_candidate))

            tmp_time = T.stop()
            time_use = time.time() - time_start
            log_run.logger.debug('贪婪得到K-Center覆盖剩余{}个点结束，用时 {:.4f} s；本次采样总共用时{:.4f} s'.format(n-1, tmp_time, time_use))
            T.start()
            result_idxs = np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]
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