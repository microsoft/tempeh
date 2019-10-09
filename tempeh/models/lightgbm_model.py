# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for classification and regression using LightGBM Framework"""

from lightgbm import LGBMRegressor, LGBMClassifier
from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class LightGBMClassifierWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for LightGBM Classifier sklearn API."""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.TREE
    lgbm_args = None

    def __init__(self):
        """Initializes the LightGBM Classifier wrapper.
        """

        lgbm_args = {} if self.lgbm_args is None else self.lgbm_args
        lgbm_args[ModelParams.RANDOM_STATE] = 777

        model = LGBMClassifier(**lgbm_args)

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
    def generate_custom_tree(cls, lgbm_args):
        """Generates a custom LightGBM Classifier wrapper.
        :param lgbm_args: parameters to pass into the LightGBM Classifier
        :type lgbm_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), {"lgbm_args": lgbm_args})

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return list(self.model.feature_importances_)


class LightGBMRegressorWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for LightGBM Regressor."""

    tasks = [Tasks.REGRESSION]
    algorithm = Algorithms.TREE
    lgbm_args = None

    def __init__(self, dataset):
        """Initializes the LightGBM Regressor wrapper.
        :param dataset: the dataset
        :type dataset: BasePerformanceDatasetWrapper
        """

        lgbm_args = {} if self.lgbm_args is None else self.lgbm_args
        lgbm_args[ModelParams.RANDOM_STATE] = 777

        model = LGBMRegressor(**lgbm_args)

        super().__init__(dataset, model)

    @classmethod
    def compatible_with_dataset(cls, dataset):
        """Checks if the model is compatible with the dataset
        :param dataset: the dataset
        :type dataset: BaseDatasetWrapper
        :rtype: bool
        """
        return dataset.task in cls.tasks and dataset.data_type == DataTypes.TABULAR

    @classmethod
    def generate_custom_tree(cls, lgbm_args):
        """Generates a custom LightGBM Regressor wrapper.
        :param lgbm_args: parameters to pass into the decision tree
        :type lgbm_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), {"lgbm_args": lgbm_args})

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return list(self.model.feature_importances_)
