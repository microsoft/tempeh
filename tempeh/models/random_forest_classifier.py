# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for random forest classifier."""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class RandomForestClassifierWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for random forest classifier"""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.TREE
    rfc_args = None

    def __init__(self):
        """Initializes the random forest classifier wrapper.
        """

        rfc_args = {} if self.rfc_args is None else self.rfc_args
        rfc_args[ModelParams.RANDOM_STATE] = 777

        model = RandomForestClassifier(**rfc_args)

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
    def generate_custom_ridge(cls, rfc_args):
        """Generates a custom decision tree wrapper.
        :param r_args: parameters to pass into the decision tree
        :type r_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), dict(rfc_args=rfc_args))

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return self.model.feature_importances_.tolist()
