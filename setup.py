# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import setuptools
import tempeh

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=tempeh.__name__,
    version=tempeh.__version__,
    author="Roman Lutz",
    author_email="romanlutz13@gmail.com",
    description="Machine Learning Performance Testing Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/romanlutz/tempeh",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "pytest",
        "tensorflow",
        "torch",
        "xgboost",
        "lightgbm"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
