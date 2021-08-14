import numpy as np
import torch
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
        self.transform = args.transform
        self.train_kwargs = args.train_kwargs
        self.test_kwargs = args.test_kwargs
        self.n_pool = len(Y)
        self.device = device

    def query(self, n):
        # *使用不同的筛选策略，不做统一规定
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.model.train()
        # *每次要取得样本、标签、对应的id
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            #  逐步记录训练的结果，包括loss、epoch等
            if batch_idx % self.args.log_interval == 0:
                tmp_time = self.T.stop()
                self.log.logger.debug('epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time is {:.4f}'.format(
                    epoch, batch_idx * len(x), len(loader_tr.dataset), 
                    100. * batch_idx / len(loader_tr), loss.item(),tmp_time
                ))
                self.args.csv_record_trloss.write_data([self.args.sampling_time, epoch, batch_idx, loss.item()])
                self.T.start()

    def train(self):
        # csv_record_trloss = self.args.csv_record_trloss

        n_epoch = self.args.epochs
        self.model = self.net().to(self.device)
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        # idxs_train记录所有被训练过的样本
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        # 读取训练集
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.transform),
                            shuffle=True, **self.train_kwargs)

        self.T.start()
        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)

    def predict(self, X, Y):
        self.T.start()

        # 读取测试集
        loader_te = DataLoader(self.handler(X, Y, transform=self.transform),
                            shuffle=False, **self.test_kwargs)

        len_testdata = len(loader_te.dataset)
        self.model.eval()
        test_loss = 0
        correct = 0
        pred_te = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)
                test_loss += F.cross_entropy(out, y, reduction='sum').item()
                # pred = out.max(1)[1]
                pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
                pred_te[idxs] = pred.cpu()

        test_loss /= len_testdata
        acc = correct / len_testdata
    
        tmp_time = self.T.stop()
        self.log.logger.info('采样次数：{}, 平均loss为：{:.4f}, 准确率为：{}/{}({:.2f}%), 预测用时：{}s'.
                            format(self.args.rd, test_loss, correct, len_testdata, 100*acc, tmp_time))
        self.args.csv_record_tracc.write_data([self.args.rd, self.args.n_budget_used, acc, test_loss])
        return  acc

    def save_results(self, file_results):
        return

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.transform),
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

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.transform),
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
        loader_te = DataLoader(self.handler(X, Y, transform=self.transform),
                            shuffle=False, **self.test_kwargs)

        self.model.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.model(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.transform),
                            shuffle=False, **self.test_kwargs)

        self.model.eval()
        embedding = torch.zeros([len(Y), self.model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

