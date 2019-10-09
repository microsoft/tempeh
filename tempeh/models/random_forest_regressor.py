# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Defines a model class for random forest regressor."""
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class RandomForestRegressorWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for random forest regressor"""

    tasks = [Tasks.REGRESSION]
    algorithm = Algorithms.TREE
    rfr_args = None

    def __init__(self):
        """Initializes the random forest regressor wrapper.
        """

        rfr_args = {} if self.rfr_args is None else self.rfr_args
        rfr_args[ModelParams.RANDOM_STATE] = 777

        model = RandomForestRegressor(**rfr_args)

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
    def generate_custom_ridge(cls, rfr_args):
        """Generates a custom decision tree wrapper.
        :param rfr_args: parameters to pass into the decision tree
        :type rfr_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), dict(rfr_args=rfr_args))

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return self.model.feature_importances_.tolist()
