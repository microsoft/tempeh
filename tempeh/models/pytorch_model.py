# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for a Pytorch DNN"""

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    print("No module named 'torch'. If you want to use pytorch with tempeh please install pytorch separately first.")

from .base_model import BaseModelWrapper

from tempeh.constants import Tasks, DataTypes, Algorithms  # noqa


class BasePytorchWrapper(BaseModelWrapper):
    """Base wrapper for Pytorch models."""

    algorithm = Algorithms.DEEP
    limit = (100000, 10000)
    pytorch_args = None

    def __init__(self, model):
        """Initializes the base pytorch model wrapper.
        :param model: the model
        :type model: model
        """
        super().__init__(None)

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return dataset.task in cls.tasks and dataset.data_type == DataTypes.TABULAR and \
            (cls.limit is None or all((cls.limit[i] is None or cls.limit[i] > dataset.size[i]
                                       for i in range(len(dataset.size)))))


class PytorchMulticlassClassifierWrapper(BasePytorchWrapper):
    """Wrapper for Pytorch multiclass classifier."""

    tasks = [Tasks.MULTICLASS]

    def __init__(self, dataset):
        """Initializes the pytorch multiclass wrapper.
        """
        super().__init__(None)

    def fit(self, X, y):
        self.create_pytorch_multiclass_classifier(X, y)
        super(PytorchMulticlassClassifierWrapper, self).fit(X, y)

    def create_pytorch_multiclass_classifier(self, X, y):
        # Get unique number of classes
        numClasses = np.unique(y).shape[0]
        # create simple (dummy) Pytorch DNN model for multiclass classification
        epochs = 12
        torch_X = torch.Tensor(X).float()
        torch_y = torch.Tensor(y).long()
        # Create network structure
        net = _common_pytorch_generator(X.shape[1], numClasses=numClasses)
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        self.model = _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


class PytorchBinaryClassifierWrapper(BasePytorchWrapper):
    """Wrapper for Pytorch multiclass classifier."""

    tasks = [Tasks.BINARY]

    def __init__(self):
        """Initializes the pytorch binary model wrapper.
        """
        super().__init__(None)

    def fit(self, X, y):
        self.create_pytorch_classifier(X, y)
        super(PytorchBinaryClassifierWrapper, self).fit(X, y)

    def create_pytorch_classifier(self, X, y):
        # create simple (dummy) Pytorch DNN model for binary classification
        epochs = 12
        torch_X = torch.Tensor(X).float()
        torch_y = torch.Tensor(y).long()
        # Create network structure
        net = _common_pytorch_generator(X.shape[1], numClasses=2)
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        self.model = _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


class PytorchRegressionWrapper(BasePytorchWrapper):
    """Wrapper for Pytorch regressor."""

    tasks = [Tasks.REGRESSION]

    def __init__(self):
        """Initializes the pytorch regression model wrapper.
        """
        super().__init__(None)

    def fit(self, X, y):
        self.create_pytorch_regressor(X, y)
        super(PytorchRegressionWrapper, self).fit(X, y)

    def create_pytorch_regressor(self, X, y):
        # create simple (dummy) Pytorch DNN model for regression
        epochs = 12
        if isinstance(X, pd.DataFrame):
            X = X.values
        torch_X = torch.Tensor(X).float()
        torch_y = torch.Tensor(y).float()
        # Create network structure
        net = _common_pytorch_generator(X.shape[1])
        # Train the model
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
        self.model = _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def _common_pytorch_generator(numCols, numClasses=None):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Apply layer normalization for stability and perf on wide variety of datasets
            # https://arxiv.org/pdf/1607.06450.pdf
            self.norm = nn.LayerNorm(numCols)
            self.fc1 = nn.Linear(numCols, 100)
            self.fc2 = nn.Dropout(p=0.2)
            if numClasses is None:
                self.fc3 = nn.Linear(100, 3)
                self.output = nn.Linear(3, 1)
            elif numClasses == 2:
                self.fc3 = nn.Linear(100, 2)
                self.output = nn.Sigmoid()
            else:
                self.fc3 = nn.Linear(100, numClasses)
                self.output = nn.Softmax()

        def forward(self, X):
            X = self.norm(X)
            X = F.relu(self.fc1(X))
            X = self.fc2(X)
            X = self.fc3(X)
            X = self.output(X)

            return X
    return Net()


def _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(torch_X)
        loss = criterion(out, torch_y)
        loss.backward()
        optimizer.step()
        print('epoch: ', epoch, ' loss: ', loss.data.item())
    return net
