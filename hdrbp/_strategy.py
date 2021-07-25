from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hdrbp._step import StepData, StepEstimationResult, StepHoldingResult, StepResult
from hdrbp._util import basic_repr, basic_str, enforce_sum_one
from hdrbp.covariance import CovarianceEstimator
from hdrbp.weight import WeightOptimizer

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class Strategy:
    def __init__(
        self,
        covariance_estimator: CovarianceEstimator,
        weight_optimizer: WeightOptimizer,
    ) -> None:
        self._covariance_estimator = covariance_estimator
        self._weight_optimizer = weight_optimizer

    @classmethod
    def from_product(
        cls,
        covariance_estimators: list[CovarianceEstimator],
        weight_optimizers: list[WeightOptimizer],
    ) -> list[Strategy]:
        strategies = []
        for covariance_estimator in covariance_estimators:
            for weight_optimizer in weight_optimizers:
                strategy = cls(covariance_estimator, weight_optimizer)
                strategies.append(strategy)

        return strategies

    @property
    def covariance_estimator(self) -> CovarianceEstimator:
        return self._covariance_estimator

    def backtest(self, data: StepData, covariances: Optional[np.ndarray] = None) -> StepResult:
        logger.debug(f"{self}: Backtesting data")

        estimation_result = self._backtest_estimation(data.estimation, covariances)
        holding_result = self._backtest_holding(data.holding, estimation_result.weights)

        result = StepResult(estimation_result, holding_result)

        return result

    def _backtest_estimation(self, data, covariances):
        if covariances is None:
            covariances = self._covariance_estimator.estimate(data.returns)

        weights = self._weight_optimizer.optimize(covariances)

        result = StepEstimationResult(
            self._covariance_estimator,
            self._weight_optimizer,
            covariances,
            weights,
        )

        return result

    def _backtest_holding(self, data, rebalance_weights):
        weights, returns = _propagate_weights(rebalance_weights, data.returns)

        result = StepHoldingResult(
            self._covariance_estimator,
            self._weight_optimizer,
            weights,
            returns,
        )

        return result


def _propagate_weights(rebalance_weights, asset_returns):
    asset_returns = np.nan_to_num(asset_returns)  # Holding data may have nans
    current_weights = rebalance_weights

    returns = []
    weights = []
    for current_asset_returns in asset_returns:
        current_return = current_weights @ current_asset_returns
        current_weights = enforce_sum_one(current_weights * (1 + current_asset_returns))

        returns.append(current_return)
        weights.append(current_weights)

    return weights, np.array(returns)
