import numpy as np
from .strategy import Strategy
# *最小置信度策略，选择最不确定的n个
class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, args_add, log, device):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args, args_add, log, device)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]