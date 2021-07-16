import logging
from typing import Optional

import pandas as pd

from hdrbp._rolling_window import RollingWindow
from hdrbp._step import Step, parse_steps
from hdrbp._strategy import Strategy
from hdrbp._util import basic_repr, basic_str
from hdrbp.metric import MetricCalculator

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class Backtester:
    def __init__(
        self,
        rolling_window: RollingWindow,
        strategies: list[Strategy],
        metric_calculators: list[MetricCalculator],
    ) -> None:
        self._rolling_window = rolling_window
        self._strategies = strategies
        self._metric_calculators = metric_calculators

        self._steps: Optional[list[Step]] = None
        self._results: Optional[pd.DataFrame] = None
        self._metrics: Optional[pd.DataFrame] = None

    def backtest(self, returns: pd.DataFrame, covariates: Optional[pd.DataFrame] = None) -> None:
        steps = self._backtest(returns, covariates)
        results = self._parse_steps(steps)
        metrics = self._calculate_metrics(results)

        self._steps = steps
        self._results = results
        self._metrics = metrics

    def _backtest(self, returns, covariates):
        logger.info(f"{self}: Backtesting")

        steps = []
        possible_step_count = self._rolling_window.count_possible_steps(returns)
        for index in range(possible_step_count):
            logger.info(f"{self}: Backtesting step {index}/{possible_step_count-1}")

            data = self._rolling_window.extract_data(index, returns, covariates)
            results = self._backtest_strategies(data)
            step = Step(index, data, results)

            steps.append(step)

        return steps

    def _backtest_strategies(self, data):
        results = []
        covariances_cache = {}
        for strategy in self._strategies:
            try:
                covariances = covariances_cache[strategy.covariance_estimator]
            except KeyError:
                result = strategy.backtest(data)

                covariances = result.estimation.covariances
                covariances_cache[strategy.covariance_estimator] = covariances
            else:
                result = strategy.backtest(data, covariances)

            results.append(result)

        return results

    def _parse_steps(self, steps):
        logger.info(f"{self}: Parsing steps")

        return parse_steps(steps)

    def _calculate_metrics(self, results):
        logger.info(f"{self}: Calculating metrics")

        metrics = (
            results.groupby(["covariance_estimator", "weight_optimizer"])
            .apply(self._calculate_group_metrics, self._metric_calculators)
            .reset_index()
        )

        return metrics

    @staticmethod
    def _calculate_group_metrics(result, calculators):
        covariance_estimator = result["covariance_estimator"].values[0]
        weight_optimizer = result["weight_optimizer"].values[0]
        logger.debug(
            f"Backtester: Calculating metrics of group "
            f"{covariance_estimator=}"
            f" and "
            f"{weight_optimizer=}"
        )

        metrics = {}
        for calculator in calculators:
            metric = calculator.calculate(result)
            metrics.update(metric)

        metrics = pd.Series(metrics)

        return metrics

    @property
    def steps(self) -> list[Step]:
        if self._steps is None:
            raise AttributeError("You must backtest before accessing the steps.")

        return self._steps

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise AttributeError("You must backtest before accessing the results.")

        return self._results

    @property
    def metrics(self) -> pd.DataFrame:
        if self._metrics is None:
            raise AttributeError("You must backtest before accessing the metrics.")

        return self._metrics
