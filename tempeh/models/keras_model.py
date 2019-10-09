# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for a Keras DNN"""

import os
import sys

from .base_model import BaseModelWrapper
from common_utils import create_keras_classifier, create_keras_multiclass_classifier, \
    create_keras_regressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))  # noqa
from constants import Tasks, DataTypes, Algorithms  # noqa


class BaseKerasWrapper(BaseModelWrapper):
    """Base wrapper for Keras models."""

    algorithm = Algorithms.DEEP
    limit = (100000, 10000)
    keras_args = None

    def __init__(self, model):
        """Initializes the base keras model wrapper.
        :param model: the model
        :type model: model
        """
        super().__init__(model)

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


class KerasMulticlassClassifierWrapper(BaseKerasWrapper):
    """Wrapper for Keras multiclass classifier."""

    tasks = [Tasks.MULTICLASS]

    def __init__(self, dataset):
        """Initializes the base model wrapper.

        :param dataset: the dataset
        :type dataset: BasePerformanceDatasetWrapper
        """
        model = create_keras_multiclass_classifier(dataset.X_train, dataset.y_train)
        super().__init__(dataset, model)


class KerasBinaryClassifierWrapper(BaseKerasWrapper):
    """Wrapper for Keras multiclass classifier."""

    tasks = [Tasks.BINARY]

    def __init__(self, dataset):
        """Initializes the base model wrapper.

        :param dataset: the dataset
        :type dataset: BasePerformanceDatasetWrapper
        """
        model = create_keras_classifier(dataset.X_train, dataset.y_train)
        super().__init__(dataset, model)


class KerasRegressionWrapper(BaseKerasWrapper):
    """Wrapper for Keras regressor."""

    tasks = [Tasks.REGRESSION]

    def __init__(self, dataset):
        """Initializes the base model wrapper.

        :param dataset: the dataset
        :type dataset: BasePerformanceDatasetWrapper
        """
        model = create_keras_regressor(dataset.X_train, dataset.y_train)
        super().__init__(dataset, model)
