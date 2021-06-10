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
        metric_calculators: Optional[list[MetricCalculator]] = None,
    ) -> None:
        self._rolling_window = rolling_window
        self._strategies = strategies
        self._metric_calculators = metric_calculators

        self._steps: Optional[list[Step]] = None
        self._result: Optional[pd.DataFrame] = None
        self._metrics: Optional[pd.DataFrame] = None

    # TODO: maybe not save steps
    def backtest(self, returns: pd.DataFrame) -> None:
        steps = self._backtest(returns)
        result = self._parse_steps(steps)
        metrics = self._calculate_metrics(result)

        self._steps = steps
        self._result = result
        self._metrics = metrics

    def _backtest(self, returns):
        logger.info(f"{self}: Backtesting")

        steps = []
        possible_step_count = self._rolling_window.count_possible_steps(returns)
        for index in range(possible_step_count):
            logger.info(f"{self}: Backtesting step {index}/{possible_step_count-1}")

            data = self._rolling_window.pack_step_data(index, returns)
            results = self._backtest_step_strategies(data)
            step = Step(index, data, results)

            steps.append(step)

        return steps

    def _backtest_step_strategies(self, data):
        results = []
        covariances_cache = {}
        for strategy in self._strategies:
            try:
                covariances = covariances_cache[strategy.covariance_estimator]
            except KeyError:
                result = strategy.backtest_step(data)

                covariances = result.estimation.covariances
                covariances_cache[strategy.covariance_estimator] = covariances
            else:
                result = strategy.backtest_step(data, covariances)

            results.append(result)

        return results

    def _parse_steps(self, steps):
        logger.info(f"{self}: Parsing steps")

        return parse_steps(steps)

    def _calculate_metrics(self, result):
        logger.info(f"{self}: Calculating metrics")

        if self._metric_calculators is None:
            return None

        metrics = (
            result.groupby(["covariance_estimator", "weight_optimizer"])
            .apply(self._calculate_group_metrics, self._metric_calculators)
            .reset_index()
        )

        return metrics

    @staticmethod
    def _calculate_group_metrics(result, calculators):
        metrics = {}
        for calculator in calculators:
            metric = calculator.calculate(result)
            metrics.update(metric)

        metrics = pd.Series(metrics)

        return metrics

    @property
    def result(self) -> pd.DataFrame:
        if self._result is None:
            raise AttributeError("You must backtest before accessing the result.")

        return self._result

    @property
    def metrics(self) -> pd.DataFrame:
        if self._metric_calculators is None:
            raise AttributeError("You must provide metric calculators to access the metrics.")

        if self._metrics is None:
            raise AttributeError("You must backtest before accessing the metrics.")

        return self._metrics
