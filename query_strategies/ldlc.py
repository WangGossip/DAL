import numpy as np
from .strategy import Strategy
# *动态学习策略，选择训练中离散程度最大的n个
class LearningDynamicLeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(LearningDynamicLeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pred_all_state = self.predict_unlabeled[idxs_unlabeled]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0] * pred_all_state.max(1)
        #* 从小到大排序，选最小的，越小说明越离散
        return idxs_unlabeled[np.argsort(U)[:n]]