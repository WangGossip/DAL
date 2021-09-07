from log import Logger
import numpy as np
import torch
import torch.nn.functional as F
def ano_log(log):
    log.logger.debug('这是其余文件测试')
    return

# 计算余弦距离
def cosine_distance(matrix1,matrix2):
        matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
        matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
        matrix1_norm = matrix1_norm[:, np.newaxis]
        matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
        matrix2_norm = matrix2_norm[:, np.newaxis]
        cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
        return cosine_distance

# 构建距离矩阵，计算欧式距离
def make_martix(args, X, device):
    T = args.timer
    log_run = args.log_run
    log_run.logger.debug('这里进行冗余矩阵计算测试')
    T.start()
    len_X = len(X)
    size_fig = X[0].numel()
    X = X.to(torch.float).reshape([len_X, size_fig])
    # M_cosdis = cosine_distance(X.numpy(), X.numpy())
    # tmp_t = T.stop()
    # log_run.logger.debug('计算冗余度用时：{:.4f} s'.format(tmp_t))
    # return
    # test
    # xt = F.normalize(X[0])
    # print(xt.shape)

    # return
    # test
    # M_dis = torch.zeros([len_X, len_X]).to(device)
    # X_norm = torch.zeros([len_X, size_fig]).to(device)
    X_norm = F.normalize(X)
    X_norm_T = X_norm.T
    M_dis = X_norm.mm(X_norm_T)
    print('shape is {}'.format(M_dis.shape))
    # tmp_t = T.stop()
    # log_run.logger.debug('准备工作完成，用时：{:.4f} s'.format(tmp_t))
    # T.start()

    # for i in range(len_X):
    #     # X_tmp = F.normalize(X[i])
    #     for j in range(i):
    #         M_dis[i][j] = X_norm[i].mm(X_norm[j].t())
        # M_dis[i] = 
        # tmp_t = T.stop()
        # log_run.logger.debug('第 {} 行计算用时： {:.4f} s'.format(i, tmp_t))
        # T.start()
        # break

    tmp_t = T.stop()
    args.log_run.logger.debug('计算冗余度用时：{:.4f} s'.format(tmp_t))