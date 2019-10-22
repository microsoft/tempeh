# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a base model class for the performance testing framework."""

from abc import abstractmethod

from tempeh.abstract import ABC  # noqa


class BaseModelWrapper(object):
    """Wrapper for models."""

    tasks = None
    algorithm = None

    def __init__(self, model):
        """Initializes the base model wrapper.
        :param model: the model
        :type model: model
        """
        self.model = model

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return True

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ExplainableMixin(ABC):
    """Mixin for explainable models."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        pass

    @property
    def true_global_importance_rank(self):
        """Gets the rank of features in decreasing order of importance.
        :returns: list of indices
        :rtype: list[int]
        """
        tuples = sorted([(relevance, index) for index, relevance
                         in enumerate(self.true_global_importance_values)], reverse=True)
        return list(list(zip(*tuples))[1])
