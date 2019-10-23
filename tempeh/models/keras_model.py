# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a model class for a Keras DNN"""

import numpy as np

try:
    import keras
    from tensorflow.keras.layers import Activation, Dense, Dropout
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError:
    print("No modules named 'keras' and 'tensorflow'. "
          "If you want to use keras and tensorflow with tempeh "
          "please install keras and tensorflow separately first.")


from .base_model import BaseModelWrapper
from tempeh.constants import Tasks, DataTypes, Algorithms  # noqa


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


def create_keras_regressor(X, y):
    # create simple (dummy) Keras DNN model for regression
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('linear'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def create_keras_classifier(X, y):
    # create simple (dummy) Keras DNN model for binary classification
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def create_keras_multiclass_classifier(X, y):
    batch_size = 128
    epochs = 12
    num_classes = len(np.unique(y))
    model = _common_model_generator(X.shape[1], num_classes)
    model.add(Dense(units=num_classes, activation=Activation('softmax')))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y, num_classes)
    model.fit(X, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y_train))
    return model


def create_dnn_classifier_unfit(feature_number):
    # create simple (dummy) Keras DNN model for binary classification
    model = _common_model_generator(feature_number)
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def _common_model_generator(feature_number, output_length=1):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(feature_number,)))
    model.add(Dropout(0.25))
    model.add(Dense(output_length, activation='relu', input_shape=(32,)))
    model.add(Dropout(0.5))
    return model
