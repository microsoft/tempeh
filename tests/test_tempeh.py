# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from .conftest import get_selected_datasets, get_selected_models


@pytest.mark.parametrize('Dataset', get_selected_datasets())
@pytest.mark.parametrize('Model', get_selected_models())
@pytest.mark.parametrize("X_format", [np.ndarray, pd.DataFrame])
@pytest.mark.parametrize("y_format", [np.ndarray, pd.Series])
def test_fit_predict(Dataset, Model, X_format, y_format):
    dataset = Dataset()
    model = Model()
    if model.compatible_with_dataset(dataset):
        X_train, X_test = dataset.get_X(format=X_format)
        y_train, _ = dataset.get_y(format=y_format)
        model.fit(X_train, y_train)
        model.predict(X_test)
