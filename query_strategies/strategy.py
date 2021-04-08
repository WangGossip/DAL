import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, args_add, device):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.args_add = args_add
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
            # todo 逐步记录训练的结果，包括loss、epoch等

    def train(self):
        n_epoch = self.args.epochs
        self.model = self.net().to(self.device)
        optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # idxs_train记录所有被训练过的样本
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        # 读取训练集
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args_add['transform']),
                            shuffle=True, **self.args_add['train_kwargs'])

        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)

    def predict(self, X, Y):
        # 读取测试集
        loader_te = DataLoader(self.handler(X, Y, transform=self.args_add['transform']),
                            shuffle=False, **self.args_add['test_kwargs'])

        self.model.eval()
        pred_te = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)

                pred = out.max(1)[1]
                pred_te[idxs] = pred.cpu()

        return pred_te

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args_add['transform']),
                            shuffle=False, **self.args_add['test_kwargs'])

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
        loader_te = DataLoader(self.handler(X, Y, transform=self.args_add['transform']),
                            shuffle=False, **self.args_add['test_kwargs'])

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
        loader_te = DataLoader(self.handler(X, Y, transform=self.args_add['transform']),
                            shuffle=False, **self.args_add['test_kwargs'])

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
        loader_te = DataLoader(self.handler(X, Y, transform=self.args_add['transform']),
                            shuffle=False, **self.args_add['test_kwargs'])

        self.model.eval()
        embedding = torch.zeros([len(Y), self.model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.model(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

