import numpy as np
import torch
import os
import time

from torch.utils.data.dataset import Dataset

# *个人编写文件库
import arguments
from query_strategies import strategy, RandomSampling, LeastConfidence, MarginSampling, EntropySampling, EntropySamplingThr, Entropy_Multi_Sampling
from function import  get_results_dir, draw_tracc, draw_samples_prop, get_init_samples, get_mnist_prop
from tools import Timer, csv_results, label_count

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
    # *参数处理部分, 这里也计时，也可以用timer
    time_start = time.time()
    T = Timer()#程序开始时间
    args.timer = T
    # 处理存储文件夹，args.out_path代表结果输出位置
    get_results_dir(args)
    # args内添加csv类，分别记录训练过程迭代的loss变化等；
    csv_record_trloss = csv_results(args, 'train_loss.csv')
    csv_record_tracc = csv_results(args, 'train_acc.csv')
    csv_record_trsample = csv_results(args, 'train_sampling.csv')#~这里存储每一次采样后当时选取的样本比例以及当前总样本比例
    args.csv_record_trloss = csv_record_trloss
    args.csv_record_tracc = csv_record_tracc
    args.csv_record_trsample = csv_record_trsample
    csv_record_trloss.write_title(['sampling_time', 'epoch', 'batch_idx', 'loss'])
    csv_record_tracc.write_title(['sampling_time', 'sampled_count', 'acc', 'loss'])
    csv_record_trsample.write_title(['sampling_time', 'current_count', 'total_count'])#分别是，采样第几次、当前采样的比例、总比例
    # logger类
    log_run = Logger(args, level=args.log_level)
    args.log_run = log_run
    # time_0 = time.time()
    # *设置日志相关参数
    name_log_file = os.path.join(args.logs_path,args.log_name)
    # 部分会常用的变量
    DATA_NAME = args.dataset
    MODEL_NAME = args.model_name
    # 随机数种子设置
    SEED=args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tmp_t = T.stop()
    log_run.logger.info('程序开始，部分基础参数处理完成，用时 {:.4f} s'.format(tmp_t))
    log_run.logger.info('使用数据集为：{}， 网络模型为：{}， epoch为：{}， batchsize为：{}， lr为：{}， 标注预算为：{}'.
                        format(DATA_NAME, MODEL_NAME, args.epochs, args.batch_size, args.lr, args.prop_budget))
    T.start()    
    # ~关于数据集参数,更新args
    # -类别实际名称
    text_labels = {
        'MNIST':['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'FashionMNIST': ['t-shirt', 'trouser', 'pullover', 'dress', 
        'coat', 'sandal', 'shirt','sneaker', 'bag', 'ankle boot']
    }
    # -类别列表
    # count_class_list = {'MNIST':10, 'FashionMNIST':10}
    count_class = len(text_labels[DATA_NAME])

    # -计算一个transform的列表
    transforms_list = {
        'MNIST':
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        'FashionMNIST':
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        'SVHN':
            [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))],
        'CIFAR10':
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    }
    tmp_transform_list = transforms_list[DATA_NAME]
    #- VGG网络需要resize为224
    if MODEL_NAME[:3] == 'VGG':
        tmp_transform_list.append(transforms.Resize(224))
    transform = transforms.Compose(tmp_transform_list)
    args.transform = transform

    # 是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # *读取数据集
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
    args.train_kwargs = train_kwargs
    args.test_kwargs = test_kwargs

    tmp_t = T.stop()
    log_run.logger.info('处理transfom、cuda等参数，读取数据集，用时 {:.4f} s'.format(tmp_t))
    T.start()

    # 数值计算部分
    times = args.times
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    n_init_pool = int(n_pool * args.prop_init)
    n_budget = int(n_pool * args.prop_budget)
    n_lb_once = (n_budget - n_init_pool) // times
    n_budget_used = 0
    log_run.logger.info('本次实验中，训练集样本数为：{}；其中初始标记数目为：{}；总预算为：{}；单次采样标记数目为：{}'.format(n_pool,n_init_pool,n_budget,n_lb_once))
    
    # 初始化标记集合
    #~ todo 修改，初始筛选策略
    idxs_lb = np.zeros(n_pool, dtype=bool) 
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    smp_idxs = get_init_samples(args, idxs_tmp, n_init_pool, Y_tr)
    idxs_lb[smp_idxs]=True
    # #test
    # props, count_lbs = get_mnist_prop(smp_idxs, Y_tr, n_init_pool, count_class)
    # print(props, count_lbs, len(smp_idxs))
    # return
    # #test
    # idxs_lb[idxs_tmp[:n_init_pool]]=True

    # 加载网络模型等
    handler = get_handler(DATA_NAME)
    net = get_net(args.model_name)
    # 筛选策略选择
    # todo 修改函数，避免多次重复
    if args.method == 'RS':
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif args.method == 'LC':
        strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif args.method == 'MS':
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif args.method == 'ES':
        strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif args.method == 'EST':
        strategy = EntropySamplingThr(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif args.method == 'EMS':
        strategy = Entropy_Multi_Sampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)

    # *训练开始
    log_run.logger.info('dataset is {},\n seed is {}, \nstrategy is {}\n'.format(DATA_NAME, SEED, type(strategy).__name__))
    # 一些参数，用于计数 首先初始化
    labels_count = label_count(times+1, n_pool)
    # *第一次训练
    rd = 0      # 记录循环采样次数，round
    args.sampling_time = rd
    args.n_budget_used = n_budget_used
    n_budget_used += n_init_pool
    strategy.train()
    acc_tmp = strategy.predict(X_te, Y_te)
    #~ todo 加一个类，改写函数、保存比例
    labels_count.write_sampling_once(smp_idxs, Y_tr, rd)
    tmp_props, tmp_total_props, tmp_count, tmp_total_count = labels_count.get_count(rd)
    log_run.logger.info('采样循环：{}， 此次循环各类别样本比例为：{}，总比例为：{}'.format(rd, tmp_props, tmp_total_props))
    samples_props = [] #用于记录每次采样时各个种类的样本比例
    samples_props_total = []
    samples_count = []
    samples_count_total = []

    acc = np.zeros(times + 1)
    acc[rd] = acc_tmp
    samples_props.append(tmp_props)
    samples_props_total.append(tmp_total_props)
    samples_count.append(tmp_count)
    samples_count_total.append(tmp_total_count)
    csv_record_trsample.write_data([rd, tmp_count, tmp_total_count])
    # 计算初次采样比例
    # tmp_props = get_mnist_prop(idxs_tmp[:n_init_pool], Y_tr, n_init_pool, count_class)
    
    # acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    # log_run.logger.info('Sampling Round {} \n testing accuracy {} \n sampling prop {} \n'.format(rd, acc[rd], tmp_props))

    for rd in range(1, times + 1):
        # *先根据筛选策略进行抽样，修改标记
        if rd != times:
            n_lb_use = n_lb_once
        else:
            n_lb_use = n_budget - n_budget_used
        n_budget_used += n_lb_use
        # args赋值
        args.sampling_time = rd
        args.n_budget_used = n_budget_used
        smp_idxs = strategy.query(n_lb_use)
        idxs_lb[smp_idxs] = True
        labels_count.write_sampling_once(smp_idxs, Y_tr, rd)
        tmp_props, tmp_total_props, tmp_count, tmp_total_count = labels_count.get_count(rd)
        samples_props.append(tmp_props)
        samples_props_total.append(tmp_total_props)
        samples_count.append(tmp_count)
        samples_count_total.append(tmp_total_count)
        csv_record_trsample.write_data([rd, tmp_count, tmp_total_count])        
        log_run.logger.info('采样循环：{}， 此次循环各类别样本比例为：{}，总比例为：{}'.format(rd, tmp_props, tmp_total_props))

        # *oracle标记环节，并训练
        strategy.update(idxs_lb)
        strategy.train()

        # *测试结果
        acc_tmp = strategy.predict(X_te, Y_te)
        acc[rd] = acc_tmp

        # acc[rd] = 1.0 * (Y_te==P_tmp).sum().item() / len(Y_te)
        # log_run.logger.info('Sampling Round {} \n testing accuracy {} \n sampling prop {} \n'.format(rd, acc[rd], tmp_props))
    
    csv_record_trloss.close()
    csv_record_tracc.close()
    csv_record_trsample.close()
    # *根据CSV画图
    T.start()
    draw_tracc(args)
    draw_samples_prop(args, samples_count, text_labels[DATA_NAME], 'samples_each_count.png')
    draw_samples_prop(args, samples_count_total, text_labels[DATA_NAME], 'samples_each_total.png')

    tmp_t = T.stop()
    log_run.logger.info('画图用时：{:.4f} s'.format(tmp_t))
    log_run.logger.info('运行log存储路径为：{}\n实验结果存储路径为：{}'.format(args.log_run.filename,args.out_path))

    # 存储训练结果：需要的是acc；loss不考虑，直接看日志；除此之外因为画图需要，需要各种比例；
    # 存一下两个数据，起始比例和预算
    sta_prop = np.zeros(2)
    sta_prop[0] = args.prop_init
    sta_prop[1] = args.prop_budget
    file_results = os.path.join(args.out_path,'{}-{}-{}-SEED{}-results.npz'.format(type(strategy).__name__, DATA_NAME, args.model_name, SEED))
    np.savez(file_results, acc=acc, sta_prop=sta_prop, samples_props=samples_props)
    time_used = time.time()-time_start
    log_run.logger.info('训练完成，本次使用采样方法为：{}；种子为{}；\n结果准确率为\n{};\n每次采样的数据比例为：\n{};共计用时：{} s'.format(type(strategy).__name__, SEED, acc, samples_props, time_used))

def test_args(args):
    print(args.save_results)
    draw_samples_prop(args)
if __name__ == '__main__':
    args = arguments.get_args()
    # test_args(args)
    main(args)

