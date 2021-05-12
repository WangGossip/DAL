import numpy as np
from .strategy import Strategy
# *取概率最大的两个之差
class RandomSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, args_add, log, device):
		super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, args_add, log, device)

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)