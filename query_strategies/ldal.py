import numpy as np
from .strategy import Strategy
# *动态学习策略，选择训练中离散程度最大的n个
class LearningDynamic(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(LearningDynamic, self).__init__(X, Y, idxs_lb, net, handler, args, device)

# todo 如何乘一个参数？
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pred_all_state = self.predict_unlabeled[idxs_unlabeled]
        U = pred_all_state.max(1)
        result_count = pred_all_state[:, 9]
        bool_equal = result_count==U
        U *= bool_equal
        #* 从小到大排序，选最小的，越小说明越离散
        return idxs_unlabeled[np.argsort(U)[:n]]