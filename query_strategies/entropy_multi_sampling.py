import numpy as np
import torch
from .strategy import Strategy
# *改进的熵策略想法
class Entropy_Multi_Sampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(Entropy_Multi_Sampling, self).__init__(X, Y, idxs_lb, net, handler, args, device)
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        probs_tmp = probs_sorted[:,:5]
        log_probs = torch.log(probs_tmp)
        U = (probs_tmp*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]