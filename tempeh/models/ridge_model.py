# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for ridge regression"""
import os
import sys
import numpy as np

from sklearn.linear_model import Ridge
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class RidgeModelWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for ridge regression."""

    tasks = [Tasks.REGRESSION]
    algorithm = Algorithms.RIDGE
    r_args = None

    def __init__(self):
        """Initializes the ridge regression model wrapper.
        :param ridge_args: args to pass to sklearn
        :type ridge_args: dict or None
        """
        r_args = self.r_args if self.r_args is not None else {ModelParams.RANDOM_STATE: 777}

        model = Ridge(**r_args)

        super().__init__(model)

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return dataset.task in cls.tasks and dataset.data_type == DataTypes.TABULAR

    @classmethod
    def generate_custom_ridge(cls, r_args):
        """Generates a custom Ridge wrapper.
        :param r_args: parameters to pass into the Ridge regression
        :type r_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ),
                    dict(r_args=r_args))

    @property
    def true_global_importance_values(self, X):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        # see adult_tabular.py
        # tl;dr this is largely a simplification but gives an idea of the importance
        # of the features

        mean = np.mean(X, axis=0, keepdims=False)
        return np.abs(self.model.coef_ * mean).tolist()
