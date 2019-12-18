# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Environment


def configure_environment(workspace, wheel_file=None):
    # collect external requirements from requirements file
    environment = Environment.from_pip_requirements(name="env", file_path="requirements.txt")

    # add private pip wheel to blob if provided
    if wheel_file:
        private_pkg = environment.add_private_pip_wheel(workspace, file_path=wheel_file)
        environment.python.conda_dependencies.add_pip_package(private_pkg)

    # add azureml-sdk to log metrics
    environment.python.conda_dependencies.add_pip_package("azureml-sdk")

    # set docker to enabled for AmlCompute
    environment.docker.enabled = True
    print("environment successfully configured")
    return environment
