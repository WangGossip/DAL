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
    print(args_add['transform'])
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
