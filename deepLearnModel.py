import numpy as np
import sqlalchemy as sqla
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy.sparse import save_npz, load_npz
from tqdm import tqdm
from collections import OrderedDict


def csr_to_tensor(csr):
    coo = csr.tocoo()
    values = torch.tensor(coo.data)
    indices = torch.tensor(np.vstack((coo.row, coo.col)))
    return torch.sparse_coo_tensor(indices, values, csr.shape)


class LinearSequentialClassifier:
    def __init__(self, layers=None, learning_rate=1e-4, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
                 train_batch_size=10, pred_batch_size=1000, device='cuda', model=None, epochs=1, weight_decay=0.005, csr_input=True):
        self.device = device
        self.layers = layers
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn(reduction='sum')

        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.csr_input = csr_input

    def data_to_loader(self, X, y):
        if self.csr_input:
            tensor_X = csr_to_tensor(X).to(self.device)
            tensor_y = torch.as_tensor(y).to(self.device)
        else:
            tensor_X = torch.as_tensor(X).to(self.device)
            tensor_y = torch.as_tensor(y).to(self.device)

        dataset = TensorDataset(tensor_X, tensor_y)
        return DataLoader(dataset, batch_size=self.train_batch_size, shuffle=False)

    def fit(self, X, y):
        if not self.model:
            self.dict = OrderedDict()
            self.dict[f'{0}'] = nn.Linear(X.shape[1], self.layers[0])

            for i in range(1, len(self.layers)):
                self.dict[f'{2 * i - 1}'] = nn.ReLU()
                self.dict[f'{2 * i}'] = nn.Linear(self.layers[i-1], self.layers[i])

            self.dict[f'{2 * (len(self.layers) - 1) - 1}'] = nn.Sigmoid()
            self.model = nn.Sequential(self.dict).to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_data_loader = self.data_to_loader(X, y)

        size = len(train_data_loader.dataset)

        for epoch in range(self.epochs):
            with tqdm(total=size, unit='rows', desc='Training Model') as pbar:
                for batch, (X, y) in enumerate(train_data_loader):
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.update(len(X))

                    if pbar.n % 500 == 0:
                        pbar.set_postfix_str(f'loss: {loss.item() / len(X):>3f}')

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1

    def predict(self, X):
        with torch.no_grad():
            if self.csr_input:
                tensor_X = csr_to_tensor(X).to(self.device)
            else:
                tensor_X = torch.as_tensor(X).to(self.device)

            return self.model(tensor_X).argmax(1).to('cpu').numpy()

    def score(self, X, y):
        pred = self.predict(X)

        size = X.shape[0]

        correct = np.count_nonzero(pred == y)

        correct /= size

        return correct

    def __str__(self):
        return 'LinearSequentialClassifier'
