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
    estimation: StepEstimationData
    holding: StepHoldingData


@dataclass
class StepEstimationData:
    dates: pd.DatetimeIndex
    assets: pd.Index
    returns: np.ndarray
    covariates: Optional[np.ndarray] = None


@dataclass
class StepHoldingData:
    dates: pd.DatetimeIndex
    assets: pd.Index
    returns: np.ndarray
    covariates: Optional[np.ndarray] = None


@dataclass
class StepResult:
    estimation: StepEstimationResult
    holding: StepHoldingResult


@dataclass
class StepEstimationResult:
    covariance_estimator: CovarianceEstimator
    weight_optimizer: WeightOptimizer
    covariances: np.ndarray
    weights: np.ndarray


@dataclass
class StepHoldingResult:
    covariance_estimator: CovarianceEstimator
    weight_optimizer: WeightOptimizer
    weights: list[np.ndarray]
    returns: np.ndarray


def parse_steps(steps: list[Step]) -> pd.DataFrame:
    logger.debug("Step: Parsing steps")

    results = _parse_steps(steps)
    results = _join_results(results)
    results = _rearrange_results(results)
    results = _clean_results(results)

    return results


def _parse_steps(steps):
    parsed_results = []
    for step in steps:
        index = step.index
        data = step.data

        logger.info(f"Step: Parsing step {index}")
        for result in step.results:
            covariance_estimator = result.estimation.covariance_estimator
            weight_optimizer = result.estimation.weight_optimizer
            logger.debug(f"Step: Parsing result {covariance_estimator=} and {weight_optimizer=}")

            estimation_parse = _parse_step_estimation(index, data.estimation, result.estimation)
            holding_parse = _parse_step_holding(index, data.holding, result.holding)

            parsed_result = pd.concat((estimation_parse, holding_parse))
            parsed_results.append(parsed_result)

    return parsed_results


def _parse_step_estimation(index, data, result):
    parse = {
        "covariance_estimator": [repr(result.covariance_estimator)],
        "weight_optimizer": [repr(result.weight_optimizer)],
        "step": [index],
        "date": [data.dates.max()],
        "is_rebalance": [True],
        "rebalance_assets": [data.assets],
        "covariances": [result.covariances],
        "rebalance_weights": [result.weights],
    }
    parse = pd.DataFrame(parse)

    return parse


def _parse_step_holding(index, data, result):
    date_count = len(data.dates)
    parse = {
        "covariance_estimator": date_count * [repr(result.covariance_estimator)],
        "weight_optimizer": date_count * [repr(result.weight_optimizer)],
        "step": date_count * [index],
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
    # merging those duplicated groups so that the joined parsed results makes
    # sense.

    logger.debug("Step: Joining parsed results")

    results = pd.concat(results)
    results = (
        results.groupby(["covariance_estimator", "weight_optimizer", "date"]).last().reset_index()
    )

    return results


def _rearrange_results(results):
    logger.debug("Step: Rearranging joined results")

    results["assets"] = results.agg(_add_assets, axis="columns")
    results["before_rebalance_assets"] = results.agg(_add_before_rebalance_assets, axis="columns")
    results["weights"] = results.agg(_add_weights, axis="columns")
    results["before_rebalance_weights"] = results.agg(_add_before_rebalance_weights, axis="columns")

    final_columns = [
        "covariance_estimator",
        "weight_optimizer",
        "step",
        "date",
        "is_rebalance",
        "assets",
        "before_rebalance_assets",
        "covariances",
        "weights",
        "before_rebalance_weights",
        "return",
    ]

    results = results[final_columns]

    return results


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


def _clean_results(results):
    type_map = {
        "covariance_estimator": "string",
        "weight_optimizer": "string",
    }
    results = results.astype(type_map)

    return results
