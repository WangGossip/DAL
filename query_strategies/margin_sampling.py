import numpy as np
from .strategy import Strategy
# *取概率最大的两个之差
class MarginSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, args_add, log, device):
        super(MarginSampling, self).__init__(X, Y, idxs_lb, net, handler, args, args_add, log, device)

    # 采样过程：先找到所有的未标记样本，对其预测，然后计算对应的差值
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1][:n]]