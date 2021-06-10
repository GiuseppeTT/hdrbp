from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from hdrbp.covariance import CovarianceEstimator
from hdrbp.weight import WeightOptimizer

logger = logging.getLogger(__name__)


@dataclass
class Step:
    index: int
    data: StepData
    results: list[StepResult]


@dataclass
class StepData:
    estimation: StepDataEstimation
    holding: StepDataHolding


@dataclass
class StepDataEstimation:
    dates: pd.DatetimeIndex
    assets: pd.Index
    returns: np.ndarray


@dataclass
class StepDataHolding:
    dates: pd.DatetimeIndex
    assets: pd.Index
    returns: np.ndarray


@dataclass
class StepResult:
    estimation: StepResultEstimation
    holding: StepResultHolding


@dataclass
class StepResultEstimation:
    covariance_estimator: CovarianceEstimator
    weight_optimizer: WeightOptimizer
    covariances: np.ndarray
    weights: np.ndarray


@dataclass
class StepResultHolding:
    covariance_estimator: CovarianceEstimator
    weight_optimizer: WeightOptimizer
    weights: list[np.ndarray]
    returns: np.ndarray


def parse_steps(steps: Optional[list[Step]]) -> pd.DataFrame:
    logger.debug("Step: Parsing steps")

    if steps is None:
        return None

    results = _parse_steps(steps)
    result = _join_results(results)
    result = _rearrange_result(result)
    result = _clean_result(result)

    return result


def _parse_steps(steps):
    logger.debug("Step: Parsing steps")

    parsed_results = []
    for step in steps:
        data = step.data
        for result in step.results:
            estimation_parse = _parse_step_estimation(data.estimation, result.estimation)
            holding_parse = _parse_step_holding(data.holding, result.holding)

            parsed_result = pd.concat((estimation_parse, holding_parse))
            parsed_results.append(parsed_result)

    return parsed_results


def _parse_step_estimation(data, result):
    parse = {
        "covariance_estimator": [repr(result.covariance_estimator)],
        "weight_optimizer": [repr(result.weight_optimizer)],
        "date": [data.dates.max()],
        "is_rebalance": [True],
        "rebalance_assets": [data.assets],
        "covariances": [result.covariances],
        "rebalance_weights": [result.weights],
    }
    parse = pd.DataFrame(parse)

    return parse


def _parse_step_holding(data, result):
    date_count = len(data.dates)
    parse = {
        "covariance_estimator": date_count * [repr(result.covariance_estimator)],
        "weight_optimizer": date_count * [repr(result.weight_optimizer)],
        "date": data.dates,
        "is_rebalance": date_count * [False],
        "holding_assets": date_count * [data.assets],
        "holding_weights": result.weights,
        "return": result.returns,
    }
    parse = pd.DataFrame(parse)

    return parse


def _join_results(results):
    # The duplicated groups come from a step's holding parse and the next
    # step's estimation parse. The pandas.core.groupby.GroupBy.last() method is
    # a working around that relies on the order that parsed results are
    # concatenated and the automatic NaN dropping. Its current effect is
    # merging those duplicated groups so that the joined parsed result makes
    # sense.
    result = pd.concat(results)
    result = (
        result.groupby(["covariance_estimator", "weight_optimizer", "date"]).last().reset_index()
    )

    return result


def _rearrange_result(result):
    result["assets"] = result.agg(_add_assets, axis="columns")
    result["before_rebalance_assets"] = result.agg(_add_before_rebalance_assets, axis="columns")
    result["weights"] = result.agg(_add_weights, axis="columns")
    result["before_rebalance_weights"] = result.agg(_add_before_rebalance_weights, axis="columns")

    final_columns = [
        "covariance_estimator",
        "weight_optimizer",
        "date",
        "is_rebalance",
        "assets",
        "before_rebalance_assets",
        "covariances",
        "weights",
        "before_rebalance_weights",
        "return",
    ]

    result = result[final_columns]

    return result


def _add_assets(row):
    if row["rebalance_assets"] is not None:
        return row["rebalance_assets"]
    else:
        return row["holding_assets"]


def _add_before_rebalance_assets(row):
    if row["rebalance_assets"] is not None:
        return row["holding_assets"]
    else:
        return None


def _add_weights(row):
    if row["rebalance_weights"] is not None:
        return row["rebalance_weights"]
    else:
        return row["holding_weights"]


def _add_before_rebalance_weights(row):
    if row["rebalance_weights"] is not None:
        return row["holding_weights"]
    else:
        return None


def _clean_result(result):
    type_map = {
        "covariance_estimator": "string",
        "weight_optimizer": "string",
    }
    result = result.astype(type_map)

    return result
