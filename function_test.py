from log import Logger
import numpy as np
import torch
def ano_log(log):
    log.logger.debug('这是其余文件测试')
    return

# 构建距离矩阵，计算欧式距离
def make_martix(args, X, device):
    T = args.timer
    T.start()
    len_X = len(X)
    size_fig = X[0].numel()
    X = X.to(device).to(torch.float).reshape([len_X, size_fig])
    
    M_dis = torch.zeros([len_X, len_X]).to(device)
    for i in range(len_X):
        X_tmp = X[i]
        for j in range(i):
            M_dis[i][j] = torch.cosine_similarity(X_tmp, X[j], dim=0)
        # M_dis[i] = 
        # break

    tmp_t = T.stop()
    args.log_run.logger.debug('用时：{:.4f} s'.format(tmp_t))