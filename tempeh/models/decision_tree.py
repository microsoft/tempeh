# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for classification using decision trees"""

from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class DecisionTreeClassifierWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for Decision Tree Classification."""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.TREE
    dtc_args = None

    def __init__(self):
        """Initializes the decision tree wrapper.
        """

        dtc_args = {} if self._dtc_args is None else self._dtc_args
        dtc_args[ModelParams.RANDOM_STATE] = 777

        model = DecisionTreeClassifier(**dtc_args)

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
    def generate_custom_tree(cls, dtc_args):
        """Generates a custom Decision Tree wrapper.
        :param dtc_args: parameters to pass into the decision tree
        :type dtc_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), {"dtc_args": dtc_args})

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return list(self._model.feature_importances_)
