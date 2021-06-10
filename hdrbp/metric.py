import logging
from abc import ABC, abstractmethod

import pandas as pd

from hdrbp._util import basic_repr, basic_str, count_dates_per_year

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class MetricCalculator(ABC):
    def calculate(self, result: pd.DataFrame) -> dict[str, float]:
        logger.debug(f"{self}: Calculating metric")

        return {repr(self): self._calculate(result)}

    @abstractmethod
    def _calculate(self, result):
        pass


class CAGR(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanReturn(MetricCalculator):
    def _calculate(self, result):
        result = result[result["return"].notna()]

        dates = result["date"]
        returns = result["return"]

        dates_per_year = count_dates_per_year(dates)
        daily_mean_return = returns.mean()

        mean_return = dates_per_year * daily_mean_return

        return mean_return


class Volatility(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class SharpeRatio(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class Turnover(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MaxDrawdown(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class ValueAtRisk(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class ExpectedShortfall(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanWeightGini(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanRiskContributionGini(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanVariance(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanCorrelation(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError


class MeanDiversificationRatio(MetricCalculator):
    def _calculate(self, result):
        raise NotImplementedError
