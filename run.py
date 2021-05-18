import numpy as np
import torch
import os
import time

# *个人编写文件库
import arguments
from query_strategies import strategy, RandomSampling, LeastConfidence, MarginSampling, EntropySampling, EntropySamplingThr, Entropy_Multi_Sampling
from function import get_mnist_prop

from torchvision import transforms
from log import Logger
from dataset import get_dataset
from dataset import get_handler
from model import get_net

# *测试区
from function_test import ano_log

# import ptvsd
# ptvsd.enable_attach(address = ('10.60.150.25', 3000))
# ptvsd.wait_for_attach()

def main(args):
    time_0 = time.time()#程序开始时间
    # *设置日志相关参数
    name_log_file = os.path.join(args.logs_path,args.log_name)
    # 日志名称精确到时分秒
    name_log_file_date = name_log_file + time.strftime(".%Y-%m-%d-%H:%M:%S", time.localtime())
    log = Logger(name_log_file_date,level='debug')
    log.logger.debug('程序开始')

    # *关于数据集参数,更新args
    DATA_NAME = args.dataset
    # todo 不同数据可能要额外计算
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
    # 随机数种子设置
    SEED=args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # *读取数据集
    time_1 = time.time()#实验开始时间(读数据)
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, args.data_path)
    # 训练、测试参数
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
    print(args_add['transform'])
    # 数值计算部分
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    n_init_pool = int(n_pool * args.prop_init)
    n_budget = int(n_pool * args.prop_budget)
    n_lb_once = int((n_budget - n_init_pool) / args.times) 
    log.logger.info('本次实验中，训练集样本数为：{}；其中初始标记数目为：{}；总预算为：{}；单次采样标记数目为：{}'.format(n_pool,n_init_pool,n_budget,n_lb_once))
    
    # 可能用到的数据
    if args.dataset == 'MNIST':
        count_class = 10
    
    # 初始化标记集合
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:n_init_pool]]=True

    # 加载网络模型等
    handler = get_handler(DATA_NAME)
    net = get_net(args.model_name)
    # 筛选策略选择
    if args.method == 'RS':
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)
    elif args.method == 'LC':
        strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)
    elif args.method == 'MS':
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)
    elif args.method == 'ES':
        strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)
    elif args.method == 'EST':
        strategy = EntropySamplingThr(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)
    elif args.method == 'EMS':
        strategy = Entropy_Multi_Sampling(X_tr, Y_tr, idxs_lb, net, handler, args, args_add, log, device)

    # *训练开始
    times = args.times
    log.logger.info('dataset is {},\n seed is {}, \nstrategy is {}\n'.format(DATA_NAME, SEED, type(strategy).__name__))
    # *第一次训练
    strategy.train()
    P = strategy.predict(X_te, Y_te)
    samples_props = [] #用于记录每次采样时各个种类的样本比例
    acc = np.zeros(times + 1)
    rd = 0      # 记录循环采样次数

    # 计算初次采样比例
    tmp_props = get_mnist_prop(idxs_tmp[:n_init_pool], Y_tr, n_init_pool, count_class)
    samples_props.append(tmp_props)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    log.logger.info('Sampling Round {} \n testing accuracy {} \n sampling prop {} \n'.format(rd, acc[rd], tmp_props))

    for rd in range(1, times + 1):
        # *先根据筛选策略进行抽样，修改标记
        smp_idxs = strategy.query(n_lb_once)
        idxs_lb[smp_idxs] = True
        tmp_props = get_mnist_prop(smp_idxs, Y_tr, n_lb_once, count_class)
        samples_props.append(tmp_props)

        # *oracle标记环节，并训练
        strategy.update(idxs_lb)
        strategy.train()

        # *测试结果
        P_tmp = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0 * (Y_te==P_tmp).sum().item() / len(Y_te)
        log.logger.info('Sampling Round {} \n testing accuracy {} \n sampling prop {} \n'.format(rd, acc[rd], tmp_props))
    # 存储训练结果：需要的是acc；loss不考虑，直接看日志；除此之外因为画图需要，需要各种比例；
    # 存一下两个数据，起始比例和预算
    sta_prop = np.zeros(2)
    sta_prop[0] = args.prop_init
    sta_prop[1] = args.prop_budget
    file_results = os.path.join(args.out_path,'{}-{}-{}-SEED{}-results.npz'.format(type(strategy).__name__, DATA_NAME, args.model_name, SEED))
    np.savez(file_results, acc=acc, sta_prop=sta_prop, samples_props=samples_props)
    log.logger.info('训练完成，本次使用采样方法为：{}；种子为{}；\n结果准确率为\n{};\n每次采样的数据比例为：\n{}'.format(type(strategy).__name__, SEED, acc, samples_props))

def test_args(args):
    print(args.save_results)
if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
    # test_args(args)
