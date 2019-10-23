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

    def __init__(self, limit=None, svm_args=None):
        """Initializes the base model wrapper.
        """
        self.limit = limit
        self.svm_args = svm_args
        if svm_args is None:
            svm_args = {"gamma": 0.001, "C": 100, "probability": True}
        svm_args[ModelParams.RANDOM_STATE] = 777

        model = svm.SVC(**svm_args)

        super().__init__(model)

    @classmethod
    def _compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return dataset.task in cls.tasks and dataset.data_type == DataTypes.TABULAR and \
            (cls.limit is None or all((cls.limit[i] is None or cls.limit[i] > dataset.size[i]
                                       for i in range(len(dataset.size)))))


class RBMSVMModelWrapper(SVMModelWrapper):
    def __init__(self):
        super().__init__(limit=(5000, 10000), svm_args=None)

    @classmethod
    def _compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return super()._compatible_with_dataset()


class LinearSVMModelWrapper(SVMModelWrapper):
    def __init__(self):
        super().__init__(limit=(10000, 10000),
                         svm_args={"kernel": "linear", "probability": True})
