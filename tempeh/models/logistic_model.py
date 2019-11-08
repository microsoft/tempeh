# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for logistic regression."""

import numpy as np

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, Algorithms, DataTypes  # noqa


class LogisticModelWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for logistic regression."""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.LOGISTIC
    l_args = None

    def __init__(self):
        """Initializes the logistic regression wrapper.
        """

        l_args = self._l_args if self._l_args is not None else {ModelParams.RANDOM_STATE: 777}
        model = LogisticRegression(**l_args)

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
    def generate_custom_logistic(cls, l_args):
        """Generates a custom Logistic Regression wrapper.
        :param l_args: parameters to pass into the logistic regression model
        :type l_args: dict
        :rtype: cls
        """

        return type("Custom" + cls.__name__, (cls, ), dict(l_args=l_args))

    @property
    def true_global_importance_values(self, dataset):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        # see adult_tabular.py
        # tl;dr this is largely a simplification but gives an idea of the importance
        # of the features

        try:
            mean = np.mean(dataset.X_test.toarray(), axis=0, keepdims=True)
        except:  # noqa: E722
            mean = np.mean(dataset.X_test, axis=0, keepdims=True)
        return np.abs(np.mean(self._model.coef_ * mean, axis=0))
