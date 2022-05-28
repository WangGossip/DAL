from cv2 import transform
import numpy as np
import torch
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# *这是基于池方法的通用框架，若仅在‘筛选策略’步骤有差异可以进行通用；
class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.T = args.timer
        self.log = args.log_run
        # self.transform = args.transform
        self.train_transform = args.train_transform
        self.test_transform = args.test_transform
        self.train_kwargs = args.train_kwargs
        self.test_kwargs = args.test_kwargs
        self.n_pool = len(Y)
        self.device = device

        self.predict_unlabeled = np.zeros((self.n_pool, 10))

    def query(self, n):
        # *使用不同的筛选策略，不做统一规定
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):

        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        # *每次要取得样本、标签、对应的id
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, _ = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            #  逐步记录训练的结果，包括loss、epoch等
            if batch_idx % self.args.log_interval == 0:
                tmp_time = self.T.stop()
                self.log.logger.debug('round: {}, epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, (Avg Loss: {:.4f} | Acc: {:.2f}%({}/{}))time is {:.4f} s'.format(
                    self.args.sampling_time, epoch, total, len(loader_tr.dataset), 
                    100. * batch_idx / len(loader_tr), loss.item(),train_loss/(batch_idx+1), 100.*correct/total,correct, total,tmp_time
                ))
                self.args.csv_record_trloss.write_data([self.args.sampling_time, epoch, batch_idx, loss.item()])
                self.T.start()
            
        # -如果遇到了这一类方法，需要记录下这样一个大数组
        if self.args.method[:2] == "LD":
            if(epoch >= self.args.jump_epoch):
                self.T.start();
                idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
                UX = self.X[idxs_unlabeled]
                UY = self.Y[idxs_unlabeled]
                loader_te = DataLoader(self.handler(UX, UY, transform = self.test_transform), shuffle=False, **self.test_kwargs)
                self.model.eval()
                with torch.no_grad():
                    for x, y, idx in loader_te:
                        x, y = x.to(self.device), y.to(self.device)
                        out, _ = self.model(x)
                        pred = out.argmax(dim=1, keepdim=True).cpu()  # get the index of the max log-probability
                        # *添加当前的预测结果
                        self.predict_unlabeled[idxs_unlabeled[idx]][pred] += 1
                tmp_time = self.T.stop();
                self.log.logger.debug("预测未标记样本用时：{:.4f} s".format(tmp_time))


    def train(self):
        # csv_record_trloss = self.args.csv_record_trloss
        time_start = time.time()
        n_epoch = self.args.epochs
        self.model = self.net.to(self.device)
        # *优化器
        if self.args.opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr = self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)
        elif self.args.opt == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)
        elif self.args.opt == 'adad':
            optimizer = optim.Adadelta(self.model.parameters(), lr = self.args.lr)        
        
        use_sch = not self.args.no_sch
        if self.args.sch == 'step':
            scheduler = StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.sch == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.tmax)
        elif self.args.sch == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.args.gamma)
        # idxs_train记录所有被训练过的样本
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        # 读取训练集
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.train_transform),
                            shuffle=True, **self.train_kwargs)

        self.T.start()
        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)
            if use_sch:
                scheduler.step()

        time_train_epoch = time.time() - time_start
        self.predict_unlabeled = np.zeros((self.n_pool, 10))
        self.log.logger.debug('此次训练用时：{:.4f} s'.format(time_train_epoch))


    def predict(self, X, Y):
        self.T.start()

        # 读取测试集
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)

        len_testdata = len(loader_te.dataset)
        self.model.eval()
        test_loss = 0
        correct = 0
        # pred_te = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)
                test_loss += F.cross_entropy(out, y, reduction='sum').item()
                # pred = out.max(1)[1]
                pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
                # pred_te[idxs] = pred.cpu()

        test_loss /= len_testdata
        acc = correct / len_testdata
    
        tmp_time = self.T.stop()
        self.log.logger.info('采样次数：{}, 平均loss为：{:.4f}, 准确率为：{}/{}({:.2f}%), 预测用时：{}s'.
                            format(self.args.sampling_time, test_loss, correct, len_testdata, 100*acc, tmp_time))
        self.args.csv_record_tracc.write_data([self.args.sampling_time, self.args.n_budget_used, acc, test_loss])
        return  acc

    def save_results(self, file_results):
        return

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)

        self.model.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        
        return probs

    def predict_prob_bmal(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)

        self.model.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        # todo 这里的变量参数有待研究
        hide_z = torch.zeros([len(Y), self.model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                hide_z[idxs] = e1.cpu()
        return probs, hide_z

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)

        self.model.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.model(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)
        time_start = time.time()
        self.model.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            self.T.start()
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.model(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
            tmp_time = self.T.stop()
            self.log.logger.info('第{}次预测结束，用时{}s'.format(i+1, tmp_time))
        time_use = time.time()-time_start
        self.log.logger.info('此次预测{}次，共计用时{}s'.format(n_drop, time_use))
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.test_transform),
                            shuffle=False, **self.test_kwargs)

        self.model.eval()
        embedding = torch.zeros([len(Y), self.model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

