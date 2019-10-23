# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for an SVM"""

from sklearn import svm
from .base_model import BaseModelWrapper

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class SVMModelWrapper(BaseModelWrapper):
    """Wrapper for SVM."""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.SVM

    def __init__(self, svm_args=None):
        """Initializes the base model wrapper.
        """
        if svm_args is None:
            svm_args = {"gamma": 0.001, "C": 100, "probability": True}
        svm_args[ModelParams.RANDOM_STATE] = 777

        model = svm.SVC(**svm_args)

        super().__init__(model)

    @classmethod
    def _compatible_with_dataset(cls, dataset, limit, svm_args):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return dataset.task in cls.tasks and dataset.data_type == DataTypes.TABULAR and \
            (limit is None or all((limit[i] is None or limit[i] > dataset.size[i]
                                   for i in range(len(dataset.size)))))


class RBMSVMModelWrapper(SVMModelWrapper):
    limit = (5000, 10000)
    svm_args = None

    def __init__(self):
        super().__init__()

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return SVMModelWrapper._compatible_with_dataset(dataset, cls.limit, cls.svm_args)


class LinearSVMModelWrapper(SVMModelWrapper):
    limit = (10000, 10000)
    svm_args = {
        "kernel": "linear",
        "probability": True
    }

    def __init__(self):
        super().__init__(svm_args=LinearSVMModelWrapper.svm_args)

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return SVMModelWrapper._compatible_with_dataset(dataset, cls.limit, cls.svm_args)
