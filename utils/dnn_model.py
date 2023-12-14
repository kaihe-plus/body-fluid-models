from time import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from sklearn.base import ClassifierMixin, BaseEstimator
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class DNNModel(ClassifierMixin, BaseEstimator):

    def __init__(self, num_features: int, num_neurons: int, num_layers: int, lr: float = 0.001, batch_size: int = 64,
                 epoch: int = 50, dropout=0., device=torch.device('cpu'), verbose=False):
        super().__init__()
        self.num_features = num_features
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.dropout = dropout
        self.device = device
        self.verbose = verbose

        self.classes_ = [0, 1]
        self.net: Optional[nn.Sequential] = None

    def fit(self, X: ndarray, y: ndarray):
        assert X.shape[-1] == self.num_features
        device = self.device

        # transform the data
        X = torch.as_tensor(X, dtype=torch.float, device=device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)

        # build deep neural network
        net = nn.Sequential(
            nn.Linear(self.num_features, self.num_neurons),
            nn.ReLU(),
        )
        for i in range(1, self.num_layers):
            net.append(
                nn.Dropout(self.dropout)
            )
            net.append(
                nn.Linear(self.num_neurons, self.num_neurons)
            )
            net.append(
                nn.ReLU()
            )
        net.append(
            nn.Linear(self.num_neurons, 2)
        )
        self.net = net = net.to(device)

        optimizer = Adam(
            net.parameters(),
            lr=self.lr
        )
        loss_fn = nn.CrossEntropyLoss()

        # train
        t0 = time()
        for epoch in range(1, self.epoch + 1):
            net.train()
            train_loss = torch.zeros([1], dtype=torch.float, device=device)

            for data, label in train_dl:
                data, label = data.to(device), label.to(device)
                output = net(data)
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach()
            train_loss /= len(train_dl)
            train_loss = train_loss.cpu().numpy()[0]

            if self.verbose:
                t = time() - t0
                print('[Epoch {:3d}, {:.0f}s] Train loss ({:.4f})'
                      .format(epoch, t, train_loss))

    def predict_proba(self, X: ndarray):
        assert self.net is not None

        device = self.device

        eval_ds = TensorDataset(
            torch.as_tensor(X, dtype=torch.float, device=device)
        )
        eval_dl = DataLoader(eval_ds, self.batch_size, shuffle=False)

        net = self.net
        with torch.no_grad():
            net.eval()
            proba = torch.concat(
                [F.softmax(net(data.to(device)), dim=-1) for data, in eval_dl],
                dim=0
            )
        proba_array = proba.cpu().numpy()
        return proba_array

    def predict(self, X: ndarray):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import accuracy_score

    my_X = np.random.randn(1000, 20)
    my_y = np.random.randint(0, 2, 1000)

    my_X_va = np.random.randn(1000, 20)
    my_y_va = np.random.randint(0, 2, 1000)

    dnn_model = DNNModel(20, 3, 16, epoch=50, verbose=True, device=torch.device('cuda'))
    dnn_model.fit(my_X, my_y)
    dnn_model.predict_proba(my_X)

    my_pred = dnn_model.predict(my_X)
    my_pred_va = dnn_model.predict(my_X_va)

    acc = accuracy_score(my_y, my_pred)
    acc_va = accuracy_score(my_y_va, my_pred_va)
    print(acc, acc_va)

    dnn_model.fit(my_X, my_y)

    pred = dnn_model.predict(my_X)
    pred_va = dnn_model.predict(my_X_va)

    acc = accuracy_score(my_y, pred)
    acc_va = accuracy_score(my_y_va, pred_va)
    print(acc, acc_va)
