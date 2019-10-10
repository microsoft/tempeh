# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from tempeh.configurations import datasets, models


# Helpers to restrict execution to single combinations via environment variables
# to parallelize execution through devops system.

def get_selected_datasets():
    selected_dataset_name = os.getenv("TEST_DATASET")
    if selected_dataset_name is None:
        print("No specific dataset selected - using all available.")
        return datasets.values()

    print("dataset '{}' selected.".format(selected_dataset_name))
    return [datasets[selected_dataset_name]]


def get_selected_models():
    selected_model_name = os.getenv("TEST_MODEL")
    print(selected_model_name)
    if selected_model_name is None:
        print("No specific model selected - using all available.")
        return models.values()

    print("model '{}' selected.".format(selected_model_name))
    return [models[selected_model_name]]
