import numpy as np
import torch
import os
import time

# *个人编写文件库
import arguments

from torchvision import transforms
from log import Logger
from dataset import get_dataset
from dataset import get_handler
from model import get_net

def main(args):
    time_0 = time.time()#程序开始时间
    # *设置日志相关参数
    name_log_file = os.path.join(args.out_path,args.log_name)
    log = Logger(name_log_file,level='debug')
    log.logger.debug('程序开始')

    # *关于数据集参数,更新args
    DATA_NAME = args.dataset
    transform_pool = {'MNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])},
                    'FashionMNIST':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])},
                    'SVHN':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])},
                    'CIFAR10':
                        {'transform':transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])}                                                        
    }

    # *读取参数，进行各项设置
    SEED=args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # *读取数据集
    time_1 = time.time()#实验开始时间(读数据)
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, args.data_path)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # 额外的参数
    args_add = transform_pool[DATA_NAME]
    args_add['train_kwargs']=train_kwargs
    args_add['test_kwargs']=test_kwargs
    log.logger.debug('当前的额外训练、测试等参数如下：\n{}\n'.format(args_add))

    # 数值计算部分
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    n_init_pool = int(n_pool * args.prop_init)
    n_budget = int(n_pool * args.prop_budget)
    n_lb_once = int((n_budget - n_init_pool) / args.times) 
    log.logger.info('本次实验中，训练集样本数为：{}；其中初始标记数目为：{}；总预算为：{}；单次采样标记数目为：{}'.format(n_pool,n_init_pool,n_budget,n_lb_once))
    
    # 初始化标记集合
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:n_init_pool]]=True

    # 加载网络模型等
    handler = get_handler(DATA_NAME)
    net = get_net(DATA_NAME)
    # strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)

# def test_args(args):
#     print(args)
#     return


if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
    # test_args(args)


# # 参数相关
# INIT_SEED = 1
# PROP_INIT_LB = 0.25
# PROP_BUDGET = 0.5
# NUM_INIT_LB = 10000
# NUM_QUERY = 1000
# NUM_ROUND = 10

# DATA_NAME = 'MNIST'
# # DATA_NAME = 'FashionMNIST'
# # DATA_NAME = 'SVHN'
# # DATA_NAME = 'CIFAR10'

# args_pool = {'MNIST':
#                 {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                  'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
#                  'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
#                  'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
#             'FashionMNIST':
#                 {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
#                  'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
#                  'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
#                  'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
#             'SVHN':
#                 {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
#                  'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
#                  'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
#                  'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
#             'CIFAR10':
#                 {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
#                  'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
#                  'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
#                  'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
#             }
# args = args_pool[DATA_NAME]

# # set seed
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.backends.cudnn.enabled = False
# # *这一部分是数据集的获取，根据参数而来
# # load dataset
# X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
# X_tr = X_tr[:40000]
# Y_tr = Y_tr[:40000]

# # start experiment
# n_pool = len(Y_tr)
# n_test = len(Y_te)
# print('number of labeled pool: {}'.format(NUM_INIT_LB))
# print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
# print('number of testing pool: {}'.format(n_test))

# # generate initial labeled pool
# idxs_lb = np.zeros(n_pool, dtype=bool)
# idxs_tmp = np.arange(n_pool)
# np.random.shuffle(idxs_tmp)
# idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# # load network
# net = get_net(DATA_NAME)
# handler = get_handler(DATA_NAME)

# strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# # strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# # strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# # strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# # strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
# # strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
# # strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
# # albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
# #              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# # strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)

# # print info
# print(DATA_NAME)
# print('SEED {}'.format(SEED))
# print(type(strategy).__name__)

# # round 0 accuracy
# strategy.train()
# P = strategy.predict(X_te, Y_te)
# acc = np.zeros(NUM_ROUND+1)
# acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
# print('Round 0\ntesting accuracy {}'.format(acc[0]))

# for rd in range(1, NUM_ROUND+1):
#     print('Round {}'.format(rd))

#     # *这一部分的步骤是随机抽取，先从未标记集合中按照随机策略筛选部分；然后将在lb集标记，并更新训练集、验证集等；
#     # query
#     q_idxs = strategy.query(NUM_QUERY)
#     idxs_lb[q_idxs] = True

#     # update
#     strategy.update(idxs_lb)
#     strategy.train()

#     # round accuracy
#     P = strategy.predict(X_te, Y_te)
#     acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
#     print('testing accuracy {}'.format(acc[rd]))

# # print results
# print('SEED {}'.format(SEED))
# print(type(strategy).__name__)
# print(acc)
