import numpy as np
import torch
from .strategy import Strategy
# *改进的熵策略想法
class EntropySamplingThr(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, args_add, log, device):
        super(EntropySamplingThr, self).__init__(X, Y, idxs_lb, net, handler, args, args_add, log, device)
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        probs_tmp = probs_sorted[:,:3]
        sum_probs_new = probs_tmp.sum(1)
        probs_new = torch.zeros(probs_tmp.shape)
        # 计算新的概率
        for i in range (0,probs_tmp.shape[0]):
            probs_new[i]=probs_tmp[i] / sum_probs_new[i]
        log_probs = torch.log(probs_new)
        U = (probs_new*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]