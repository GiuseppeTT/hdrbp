import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from hdrbp._util import (
    basic_repr,
    basic_str,
    compute_correlation,
    compute_diversification_ratio,
    compute_drawdowns,
    compute_gini,
    compute_prices,
    compute_risk_contributions,
    compute_turnover,
    compute_variance,
    count_dates_per_year,
    count_years,
)

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class MetricCalculator(ABC):
    @property
    def name(self):
        return repr(self)

    @abstractmethod
    def calculate(self, result: pd.DataFrame) -> float:
        pass


class GeometricMeanReturn(MetricCalculator):
    def __init__(self, annualized: bool = False) -> None:
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        log_returns = np.log1p(returns)
        mean_log_return = np.mean(log_returns)

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            dates_per_year = count_dates_per_year(dates)
            mean_log_return = dates_per_year * mean_log_return

        geometric_mean_return = np.expm1(mean_log_return)

        return geometric_mean_return


class MeanReturn(MetricCalculator):
    def __init__(self, annualized: bool = False) -> None:
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        mean_return = np.mean(returns)

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            dates_per_year = count_dates_per_year(dates)
            mean_return = dates_per_year * mean_return

        return mean_return


class Volatility(MetricCalculator):
    def __init__(self, annualized: bool = False) -> None:
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        volatility = np.std(returns)

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            dates_per_year = count_dates_per_year(dates)
            volatility = np.sqrt(dates_per_year) * volatility

        return volatility


class SharpeRatio(MetricCalculator):
    def __init__(self, annualized: bool = False) -> None:
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        mean_return = MeanReturn(self._annualized).calculate(result)
        volatility = Volatility(self._annualized).calculate(result)

        shape_ratio = mean_return / volatility

        return shape_ratio


class MeanTurnover(MetricCalculator):
    def __init__(self, annualized: bool = False) -> None:
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)
        result = _filter_rebalance_dates(result)

        turnovers = result.apply(
            lambda df: compute_turnover(
                df["before_rebalance_assets"],
                df["before_rebalance_weights"],
                df["assets"],
                df["weights"],
            ),
            axis="columns",
        )
        turnovers = turnovers.values

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            year_count = count_years(dates)
            mean_turnover = np.sum(turnovers) / year_count
        else:
            mean_turnover = np.mean(turnovers)

        return mean_turnover


class MaxDrawdown(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        prices = compute_prices(returns)
        drawdowns = compute_drawdowns(prices)
        max_drawdown = np.max(drawdowns)

        return max_drawdown


class ValueAtRisk(MetricCalculator):
    def __init__(self, probability: float = 0.95, annualized: bool = False) -> None:
        self._probability = probability
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        value_at_risk = np.quantile(returns, 1 - self._probability)

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            dates_per_year = count_dates_per_year(dates)
            value_at_risk = np.sqrt(dates_per_year) * value_at_risk

        return value_at_risk


class ExpectedShortfall(MetricCalculator):
    def __init__(self, probability: float = 0.95, annualized: bool = False) -> None:
        self._probability = probability
        self._annualized = annualized

    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_valid_returns(result)

        returns = result["return"].values
        cut_off = np.quantile(returns, 1 - self._probability)
        cut_off_returns = returns[returns <= cut_off]
        expected_shortfall = np.mean(cut_off_returns)

        if self._annualized:
            dates = pd.to_datetime(result["date"].values)
            dates_per_year = count_dates_per_year(dates)
            expected_shortfall = np.sqrt(dates_per_year) * expected_shortfall

        return expected_shortfall


class MeanWeightGini(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_rebalance_dates(result)

        weights = result["weights"]
        weights_gini = weights.apply(compute_gini)
        weights_gini = weights_gini.values
        mean_weight_gini = np.mean(weights_gini)

        return mean_weight_gini


class MeanRiskContributionGini(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_rebalance_dates(result)

        risk_contributions = result.apply(
            lambda df: compute_risk_contributions(df["covariances"], df["weights"]),
            axis="columns",
        )
        risk_contributions_gini = risk_contributions.apply(compute_gini)
        risk_contributions_gini = risk_contributions_gini.values
        mean_risk_contribution_gini = np.mean(risk_contributions_gini)

        return mean_risk_contribution_gini


class MeanVariance(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_rebalance_dates(result)

        variances = result.apply(
            lambda df: compute_variance(df["covariances"], df["weights"]),
            axis="columns",
        )
        variances = variances.values
        mean_variance = np.mean(variances)

        return mean_variance


class MeanCorrelation(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_rebalance_dates(result)

        correlations = result.apply(
            lambda df: compute_correlation(df["covariances"], df["weights"]),
            axis="columns",
        )
        correlations = correlations.values
        mean_correlation = np.mean(correlations)

        return mean_correlation


class MeanDiversificationRatio(MetricCalculator):
    def calculate(self, result: pd.DataFrame) -> float:
        logger.debug(f"{self}: Calculating metric")

        result = _filter_rebalance_dates(result)

        diversification_ratios = result.apply(
            lambda df: compute_diversification_ratio(df["covariances"], df["weights"]),
            axis="columns",
        )
        diversification_ratios = diversification_ratios.values
        mean_diversification_ratio = np.mean(diversification_ratios)

        return mean_diversification_ratio


def _filter_valid_returns(result):
    is_valid_returns = result["return"].notna()
    filtered_result = result[is_valid_returns]

    return filtered_result


def _filter_rebalance_dates(result):
    is_rebalance = result["is_rebalance"].values
    filtered_result = result[is_rebalance]

    return filtered_result


def calculate_portfolio_metrics(
    result: pd.DataFrame,
    calculators: list[MetricCalculator],
) -> pd.Series:
    covariance_estimator = result["covariance_estimator"].values[0]
    weight_optimizer = result["weight_optimizer"].values[0]
    logger.debug(
        f"Backtester: Calculating metrics of portfolio "
        f"{covariance_estimator=}"
        f" + "
        f"{weight_optimizer=}"
    )

    metrics = {}
    for calculator in calculators:
        name = calculator.name
        metric = calculator.calculate(result)
        metrics.update({name: metric})

    metrics = pd.Series(metrics)

    return metrics
