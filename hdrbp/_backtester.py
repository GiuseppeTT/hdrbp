import logging
from typing import Optional

import pandas as pd

from hdrbp._portfolio import Portfolio
from hdrbp._rolling_window import RollingWindow
from hdrbp._step import Step, parse_steps
from hdrbp._util import basic_repr, basic_str
from hdrbp.metric import MetricCalculator, calculate_portfolio_metrics

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class Backtester:
    def __init__(
        self,
        rolling_window: RollingWindow,
        portfolios: list[Portfolio],
        metric_calculators: list[MetricCalculator],
    ) -> None:
        self._rolling_window = rolling_window
        self._portfolios = portfolios
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
            results = self._backtest_portfolios(data)
            step = Step(index, data, results)

            steps.append(step)

        return steps

    def _backtest_portfolios(self, data):
        results = []
        covariances_cache = {}
        for portfolio in self._portfolios:
            try:
                covariances = covariances_cache[portfolio.covariance_estimator]
            except KeyError:
                result = portfolio.backtest(data)

                covariances = result.estimation.covariances
                covariances_cache[portfolio.covariance_estimator] = covariances
            else:
                result = portfolio.backtest(data, covariances)

            results.append(result)

        return results

    def _parse_steps(self, steps):
        logger.info(f"{self}: Parsing steps")

        return parse_steps(steps)

    def _calculate_metrics(self, results):
        logger.info(f"{self}: Calculating metrics")

        metrics = (
            results.groupby(["covariance_estimator", "weight_optimizer"])
            .apply(calculate_portfolio_metrics, self._metric_calculators)
            .reset_index()
        )

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
