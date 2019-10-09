# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines an ABC that can be used regardless of Python version."""

import six

if six.PY3:
    from abc import ABC  # noqa
else:
    from abc import ABCMeta  # noqa

    class ABC(object):
        __metaclass__ = ABCMeta
