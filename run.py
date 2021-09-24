import numpy as np
import torch
import os
import time

from torch.utils.data.dataset import Dataset

# *个人编写文件库
import arguments
from query_strategies import strategy, RandomSampling, LeastConfidence, MarginSampling, EntropySampling, EntropySamplingThr, Entropy_Multi_Sampling, BMAL, BMALSampling, DRAL
from function import  get_results_dir, draw_tracc, draw_samples_prop, get_init_samples, get_mnist_prop, get_hms_time, draw_acc_loss_all, draw_samples_prop_all
from tools import Timer, csv_results, label_count

from torchvision import transforms
from log import Logger
from dataset import get_dataset
from dataset import get_handler
from model import get_net

# *测试区
from function_test import ano_log, make_martix


def main(args):
    # *使用不同初始种子循环多次，完整框架
    # -初始的参数处理部分，开始使用Timer计时
    time_start = time.time()
    T = Timer()#程序开始时间
    args.timer = T
    repeat_times = args.repeat_times #整体的循环次数

    # 处理存储文件夹，args.out_path代表结果输出位置
    get_results_dir(args)
    # ~三个列表存储的是每一次AL学习过程后对应类
    list_train_loss = []
    list_train_acc = []
    list_train_sampling = []

    # 一些后面用到的内容
    samples_count_all = []
    samples_count_total_all = []
    str_train_result = []
    times_sampling_all = []
    n_budget_used_all = []
    # logger类
    log_run = Logger(args, level=args.log_level)
    args.log_run = log_run

    # 部分会常用的变量
    DATA_NAME = args.dataset
    MODEL_NAME = args.model_name
    # 随机数种子设置
    #~ 一共需要5个种子，先来一个初始种子，然后再根据这个来随机五次，作为每一次循环的种子
    method_seed = args.method_seed 
    if method_seed == 'time':
        seed_global = int(time_start)
    elif method_seed == 'const':
        seed_global = args.seed
    np.random.seed(seed_global)
    seed_list = np.random.randint(1, 1000, repeat_times)
    args.seed_list = seed_list

    tmp_t = T.stop()
    log_run.logger.info('程序开始，部分基础参数预处理完成，用时 {:.4f} s'.format(tmp_t))
    log_run.logger.info('''使用数据集为：{}， 网络模型为：{}， epoch为：{}， batchsize为：{}， lr为：{}， 标注预算为：{}，\n
                        需进行{}次实验，随机种子列表为：{}'''.
                        format(DATA_NAME, MODEL_NAME, args.epochs, args.batch_size, args.lr, args.prop_budget, repeat_times, seed_list))
    
    T.start()    
    # ~关于数据集参数,更新args
    # -类别实际名称
    text_labels = {
        'MNIST':['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'FashionMNIST': ['t-shirt', 'trouser', 'pullover', 'dress', 
        'coat', 'sandal', 'shirt','sneaker', 'bag', 'ankle boot']
    }

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

    # -关于筛选预算，其实只需要知道：初始预算、每次筛选样本数目、最终限制的样本数目
    method_budget = args.method_budget
    n_pool = len(Y_tr)
    if method_budget == 'prop':
        times = args.times#采样次数
        n_init_pool = int(n_pool * args.prop_init)
        n_budget = int(n_pool * args.prop_budget)
        n_lb_once = (n_budget - n_init_pool) // times
        acc_expected = 1 #~预算模式就把全部预算用满为止，不考虑acc
    elif method_budget == 'num':
        n_init_pool = args.budget_init
        n_lb_once = args.budget_once
        n_budget = n_pool #~这种模式对预算不做限制，仅看准确率合适达标
        acc_expected = args.acc_expected
        times = (n_budget - n_init_pool) // n_lb_once
        if (n_budget - n_init_pool) % n_lb_once != 0:
            times += 1 #考虑最后一次采样

    log_run.logger.info('''本次实验中，训练集样本数为：{}；其中初始标记数目为：{}；总预算为：{}；单次采样标记数目为：{}；预期准确率为：{}'''
    .format(n_pool, n_init_pool, n_budget, n_lb_once, acc_expected))


    for repeat_id in range(repeat_times):
        n_budget_used = 0
        repeat_round = repeat_id + 1
        time_start_round = time.time()
        # -从头进行一次AL迭代
        # args内添加csv类，分别记录训练过程迭代的loss变化等；
        csv_tmp_trloss = 'train_loss_round{}.csv'.format(repeat_round)
        csv_tmp_tracc = 'train_acc_round{}.csv'.format(repeat_round)
        csv_tmp_trsample = 'train_sampling_round{}.csv'.format(repeat_round)
        csv_record_trloss = csv_results(args, csv_tmp_trloss)
        csv_record_tracc = csv_results(args, csv_tmp_tracc)
        csv_record_trsample = csv_results(args, csv_tmp_trsample)#~这里存储每一次采样后当时选取的样本比例以及当前总样本比例
        args.csv_record_trloss = csv_record_trloss
        args.csv_record_tracc = csv_record_tracc
        args.csv_record_trsample = csv_record_trsample
        csv_record_trloss.write_title(['sampling_time', 'epoch', 'batch_idx', 'loss'])
        csv_record_tracc.write_title(['sampling_time', 'sampled_count', 'acc', 'loss'])
        csv_record_trsample.write_title(['sampling_time', 'current_count', 'total_count'])#分别是，采样第几次、当前采样的比例、总比例
        
        # -当次实验种子
        SEED = seed_list[repeat_id]
        args.seed = SEED
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        # 初始化标记集合
        idxs_lb = np.zeros(n_pool, dtype=bool) 
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        smp_idxs = get_init_samples(args, idxs_tmp, n_init_pool, Y_tr)
        idxs_lb[smp_idxs]=True        

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
        elif args.method == 'BMAL':
            strategy = BMAL(X_tr, Y_tr, idxs_lb, net, handler, args, device)
        elif args.method == 'BMALS':
            strategy = BMALSampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)
        elif args.method == 'DRAL':
            strategy = DRAL(X_tr, Y_tr, idxs_lb, net, handler, args, device)
        
        # *训练开始
        log_run.logger.info('dataset is {},\n seed is {}, \nstrategy is {}\n'.format(DATA_NAME, SEED, type(strategy).__name__))
        # 一些参数，用于计数 首先初始化
        labels_count = label_count(times+1, n_pool)
        # *第一次训练
        rd = 0      # 记录循环采样次数，round
        args.sampling_time = rd
        n_budget_used += n_init_pool
        args.n_budget_used = n_budget_used
        strategy.train()
        acc_tmp = strategy.predict(X_te, Y_te)
        #~ todo 加一个类，改写函数、保存比例
        labels_count.write_sampling_once(smp_idxs, Y_tr, rd)
        tmp_props, tmp_total_props, tmp_count, tmp_total_count = labels_count.get_count(rd)
        log_run.logger.info('采样循环：{}， 此次循环各类别样本采样数为：{}，占总体比例为：{}'.format(rd, tmp_count, tmp_total_props))
        samples_props = [] #用于记录每次采样时各个种类的样本比例
        samples_props_total = []
        samples_count = []
        samples_count_total = []

        acc = []
        acc.append(acc_tmp)
        samples_props.append(tmp_props)
        samples_props_total.append(tmp_total_props)
        samples_count.append(tmp_count)
        samples_count_total.append(tmp_total_count)
        csv_record_trsample.write_data([rd, tmp_count, tmp_total_count])

        # ~判断迭代终止条件：达到预算，或者准确率达到预期，即不满足其中一个即可终止
        while n_budget_used < n_budget and acc_tmp < acc_expected:
            rd = rd + 1
            # *先根据筛选策略进行抽样，修改标记
            # ~ n_lb_use代表当前迭代要使用的预算，要考虑每次预算以及剩余
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
            log_run.logger.info('采样循环：{}， 此次循环各类别样本采样数为：{}，占总体比例为：{}'.format(rd, tmp_count, tmp_total_props))

            # *oracle标记环节，并训练
            strategy.update(idxs_lb)
            strategy.train()

            # *测试结果
            acc_tmp = strategy.predict(X_te, Y_te)
            acc.append(acc_tmp)

        csv_record_trloss.close()
        csv_record_tracc.close()
        csv_record_trsample.close()

        samples_count_all.append(samples_count)
        samples_count_total_all.append(samples_count_total)
        times_sampling_all.append(rd+1)#~记录总采样次数，初始化一次也算在内
        n_budget_used_all.append(n_budget_used)   
        
        list_train_loss.append(csv_record_trloss)
        list_train_acc.append(csv_record_tracc)
        list_train_sampling.append(csv_record_trsample)

        time_use_round = time.time() - time_start_round
        h, m, s = get_hms_time(time_use_round)

        str_train_reslut_tmp = '第{}（总共{}次）次实验结束，实验用时：{}h {}min {:.4f}s，本次实验使用种子为：{}，最终使用预算为：{}，实验预测结果为：\n{}'.format(repeat_round, repeat_times, h, m, s, SEED, n_budget_used, acc)
        str_train_result.append(str_train_reslut_tmp)
        log_run.logger.info(str_train_reslut_tmp)

    # * 画图，展示结果部分
    T.start()
    # - 图1，每行展示3张图片，分别是每一次的loss、acc变化情况，横坐标是选取的样本的数量；
    draw_acc_loss_all(args, list_train_acc, times_sampling_all)
    # -总体画图，平均的样本采样情况，然后展示最终结果
    draw_samples_prop_all(args, samples_count_all, text_labels[DATA_NAME], times_sampling_all, 'samples_each_count.png')
    draw_samples_prop_all(args, samples_count_total_all, text_labels[DATA_NAME], times_sampling_all, 'samples_each_total.png')

    tmp_t = T.stop()
    log_run.logger.info('画图用时：{:.4f} s'.format(tmp_t))
    time_used = time.time()-time_start
    h, m, s = get_hms_time(time_used)
    log_run.logger.info('运行log存储路径为：{};实验结果存储路径为：{}'.format(args.log_run.filename,args.out_path))
    log_run.logger.info('训练完成，本次使用采样方法为：{}；\n实验结果为：'.format(type(strategy).__name__))
    for str in str_train_result:
        log_run.logger.info(str)
    log_run.logger.info('实验最终平均预算为：{}；共计用时：{}h {}min {:.4f}s'.format(np.mean(n_budget_used_all),h, m, s))


    # #test
    # props, count_lbs = get_mnist_prop(smp_idxs, Y_tr, n_init_pool, count_class)
    # print(props, count_lbs, len(smp_idxs))
    # return
    # #test
    # idxs_lb[idxs_tmp[:n_init_pool]]=True

def test_args(args):
    print(args.save_results)
    draw_samples_prop(args)
if __name__ == '__main__':
    args = arguments.get_args()
    # test_args(args)
    main(args)

