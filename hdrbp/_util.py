from __future__ import annotations

import inspect
import sys
from typing import Any

import numpy as np
import pandas as pd

FLOAT_RESOLUTION = sys.float_info.epsilon


def basic_repr(cls: type) -> type:
    def __repr__(self) -> str:
        init_signature = inspect.signature(self.__class__)
        init_arguments = init_signature.parameters.keys()
        attributes = self.__dict__

        init_pairs = {}
        for init_argument in init_arguments:
            if init_argument in attributes:
                init_pairs[init_argument] = attributes[init_argument]
            elif f"_{init_argument}" in attributes:
                init_pairs[init_argument] = attributes[f"_{init_argument}"]
            else:
                raise AttributeError("Init argument not in attributes.")

        class_name = self.__class__.__name__
        init_calls = [f"{argument}={repr(value)}" for argument, value in init_pairs.items()]
        init_call = ", ".join(init_calls)

        representation = f"{class_name}({init_call})"

        return representation

    setattr(cls, "__repr__", __repr__)

    return cls


def basic_str(cls: type) -> type:
    def __str__(self) -> str:
        return self.__class__.__name__

    setattr(cls, "__str__", __str__)

    return cls


def build_covariances(correlations: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    return correlations * volatilities[:, None] * volatilities[None, :]


def count_dates_per_year(dates: pd.Series) -> float:
    date_count = dates.size
    year_count = count_years(dates)

    dates_per_year = date_count / year_count

    return dates_per_year


def count_years(dates: pd.Series) -> float:
    period_size = dates.max() - dates.min()
    year_size = np.timedelta64(1, "Y")

    year_count = period_size / year_size

    return year_count


def demean(array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    return array - np.mean(array, *args, **kwargs)


def enforce_sum_one(array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    return array / np.sum(array, *args, **kwargs)


def extract_correlations(covariances: np.ndarray) -> np.ndarray:
    volatilities = extract_volatilities(covariances)

    correlations = covariances / volatilities[:, None] / volatilities[None, :]
    correlations = np.clip(correlations, -1, 1)

    return correlations


def extract_upper_elements(array: np.ndarray) -> np.ndarray:
    upper_element_indices = np.triu_indices_from(array, 1)
    upper_elements = array[upper_element_indices]

    return upper_elements


def extract_volatilities(covariances: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(covariances))
