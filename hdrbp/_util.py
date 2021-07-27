from __future__ import annotations

import inspect
import sys
from typing import Any

import numpy as np
import pandas as pd

FLOAT_RESOLUTION = sys.float_info.epsilon
ERROR_TOLERANCE = np.sqrt(FLOAT_RESOLUTION)  # Roughly 10 ** -8
CVXOPT_OPTIONS = {"show_progress": False}


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


def build_covariances(standard_deviations: np.ndarray, correlations: np.ndarray) -> np.ndarray:
    return correlations * standard_deviations[:, np.newaxis] * standard_deviations[np.newaxis, :]


def compute_correlation(covariances: np.ndarray, weights: np.ndarray) -> float:
    correlations = extract_correlations(covariances)
    correlation = weights @ correlations @ weights

    return correlation


def compute_diversification_ratio(covariances: np.ndarray, weights: np.ndarray) -> float:
    volatilities = extract_standard_deviations(covariances)
    naive_volatility = weights @ volatilities
    volatility = np.sqrt(weights @ covariances @ weights)

    diversification_ratio = naive_volatility / volatility

    return diversification_ratio


def compute_drawdowns(prices: np.ndarray) -> np.ndarray:
    return (np.maximum.accumulate(prices) - prices) / np.maximum.accumulate(prices)


def compute_gini(array: np.ndarray) -> float:
    array = np.sort(array)
    size = array.size
    indexes = np.arange(1, size + 1)

    gini = 2 / size * np.sum(indexes * array) / np.sum(array) - (size + 1) / size

    return gini


def compute_prices(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + returns)


def compute_risk_contributions(covariances: np.ndarray, weights: np.ndarray) -> float:
    risk_contributions = weights * (covariances @ weights)
    risk_contributions = enforce_unitary_sum(risk_contributions)

    return risk_contributions


def compute_turnover(
    before_rebalance_assets: pd.Index,
    before_rebalance_weights: np.ndarray,
    rebalance_assets: pd.Index,
    rebalance_weights: np.ndarray,
) -> float:
    before_rebalance_weights = pd.Series(before_rebalance_weights, before_rebalance_assets)
    rebalance_weights = pd.Series(rebalance_weights, rebalance_assets)

    all_assets = before_rebalance_assets.union(rebalance_assets)
    before_rebalance_weights = before_rebalance_weights.reindex(  # type: ignore
        all_assets, fill_value=0
    )
    rebalance_weights = rebalance_weights.reindex(all_assets, fill_value=0)  # type: ignore

    turnover = np.sum(np.abs(rebalance_weights - before_rebalance_weights))

    return turnover


def compute_variance(covariances: np.ndarray, weights: np.ndarray) -> float:
    return weights @ covariances @ weights


def count_dates_per_year(dates: pd.DatetimeIndex) -> float:
    date_count = dates.size
    year_count = count_years(dates)

    dates_per_year = date_count / year_count

    return dates_per_year


def count_digits(number: int) -> int:
    return int(np.floor(np.log10(number)))


def count_years(dates: pd.DatetimeIndex) -> float:
    period_size = dates.max() - dates.min()
    year_size = np.timedelta64(1, "Y")

    year_count = period_size / year_size

    return year_count


def demean(array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    return array - np.mean(array, *args, **kwargs)


def enforce_unitary_sum(array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    return array / np.sum(array, *args, **kwargs)


def extract_correlations(covariances: np.ndarray) -> np.ndarray:
    standard_deviations = extract_standard_deviations(covariances)

    correlations = (
        covariances / standard_deviations[:, np.newaxis] / standard_deviations[np.newaxis, :]
    )
    correlations = np.clip(correlations, -1, 1)

    return correlations


def extract_solution(optimization_results: dict) -> np.ndarray:
    solution = optimization_results["x"]
    solution = np.array(solution)
    solution = solution.flatten()

    return solution


def extract_standard_deviations(covariances: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(covariances))


def extract_upper_elements(array: np.ndarray) -> np.ndarray:
    upper_element_indices = np.triu_indices_from(array, 1)
    upper_elements = array[upper_element_indices]

    return upper_elements
