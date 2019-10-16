# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
# TODO: remove shap dependency
import shap

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def data_init(blob_name, file_name):
    """Init connection to blob and download file

    :param blob_name: name of target blob
    :type blob_name: str
    :param file_name: name the file will take
    :type file_name: str
    """

    urlretrieve("https://petrstoragedwqopnde.blob.core.windows.net/uci-datasets/" + blob_name,
                file_name)
    return file_name


def create_df(data, splitter, features=None):
    """ Creates a pandas dataframe from the data

    :param data: The data to be turned into a dataframe
    :type data: Read file
    :param splitter: str to split on
    :type splitter: str
    :param features: List of the features
    :type features: An array
    """
    if features is None:
        first_row = True
    else:
        first_row = False
        df = {j: [] for j in features}
    for i in data:
        parts = i[:-1].split(splitter)
        if first_row:
            features = [j[1:-1] for j in parts]
            df = {j: [] for j in features}
            first_row = False
        else:
            for j in range(len(features)):
                df[features[j]].append(parts[j])
    return pd.DataFrame(df)


def bank_data_parser():
    """Data set cleaner for bank data"""

    full_file_path = data_init("bank_marketing/bank-full.csv", "bank-full.csv")
    data = open(full_file_path, "r")
    df = create_df(data, ';')
    categories = ["job", "marital", "education", "contact", "month", "poutcome"]
    binary = ["default", "housing", "loan", "y"]
    df[categories] = df[categories].apply(lambda x: x[1:-1])
    df = pd.get_dummies(df, columns=categories)
    df[binary] = df[binary].eq('"yes"').mul(1)
    return df


def bank_data_additional_parser():
    """Data set cleaner for additional bank data"""

    full_file_path = data_init("bank_marketing/bank-additional-full.csv",
                               "bank-additional-full.csv")
    data = open(full_file_path, "r")
    df = create_df(data, ';')
    categories = ["job", "marital", "education", "default", "housing",
                  "loan", "contact", "month", "day_of_week", "poutcome"]
    binary = ["y"]
    df[categories] = df[categories].apply(lambda x: x[1:-1])
    df = pd.get_dummies(df, columns=categories)
    df[binary] = df[binary].eq('"yes"').mul(1)
    return df


def car_eval_parser():
    """Data set cleaner for car evaluation data"""

    full_file_path = data_init("car_eval/car.data", "car.data")
    data = open(full_file_path, "r")
    features = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "CAR"]
    df = create_df(data, ',', features=features)
    df = pd.get_dummies(df, columns=features[:-1])
    df["CAR"] = df["CAR"].astype("category")
    df["CAR"] = df["CAR"].cat.codes
    return df


def adult_data_parser():
    adult_dataset = shap.datasets.adult()
    # hack to put target column into the same dataframe
    return pd.concat((adult_dataset[0], pd.DataFrame(adult_dataset[1], columns=["y"])), axis=1)