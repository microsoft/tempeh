# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for classification and regression using XGBoost Framework"""

import logging
logger = logging.getLogger(__file__)

try:
    from xgboost import XGBRegressor, XGBClassifier
except:
    logger.debug("No module named 'xgboost'. If you want to use xgboost with tempeh please "
                 "install xgboost separately first.")

from .base_model import BaseModelWrapper, ExplainableMixin

from tempeh.constants import ModelParams, Tasks, DataTypes, Algorithms  # noqa


class XGBoostClassifierWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for XGBoost Classifier sklearn API."""

    tasks = [Tasks.BINARY, Tasks.MULTICLASS]
    algorithm = Algorithms.TREE
    xgb_args = None

    def __init__(self, dataset):
        """Initializes the XGBoost Classifier wrapper.
        :param dataset: the dataset
        :type dataset: BasePerformanceDatasetWrapper
        """

        xgb_args = {} if self._xgb_args is None else self._xgb_args
        xgb_args[ModelParams.RANDOM_STATE] = 777

        model = XGBClassifier(**xgb_args)

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
    def generate_custom_tree(cls, xgb_args):
        """Generates a custom XGBoost Classifier wrapper.
        :param xgb_args: parameters to pass into the XGBoost Classifier
        :type xgb_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), {"xgb_args": xgb_args})

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return list(self._model.feature_importances_)


class XGBoostRegressorWrapper(BaseModelWrapper, ExplainableMixin):
    """Wrapper for XGBoost Regressor."""

    tasks = [Tasks.REGRESSION]
    algorithm = Algorithms.TREE
    xgb_args = None

    def __init__(self):
        """Initializes the XGBoost Regressor wrapper.
        """

        xgb_args = {} if self._xgb_args is None else self._xgb_args
        xgb_args[ModelParams.RANDOM_STATE] = 777

        model = XGBRegressor(**xgb_args)

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
    def generate_custom_tree(cls, xgb_args):
        """Generates a custom XGBoost Regressor wrapper.
        :param xgb_args: parameters to pass into the decision tree
        :type xgb_args: dict
        :rtype: cls
        """
        return type("Custom" + cls.__name__, (cls, ), {"xgb_args": xgb_args})

    @property
    def true_global_importance_values(self):
        """Gets the feature importances.
        :returns: a list of global feature importances
        :rtype: list[float]
        """
        return list(self._model.feature_importances_)
