import logging

import numpy as np
import pandas as pd

from hdrbp import Backtester, RollingWindow, Strategy
from hdrbp.asset import LiquidAsset
from hdrbp.covariance import (
    EqualCorrelation,
    EqualVariance,
    LinearShrinkage,
    RiskMetrics1994,
    RiskMetrics2006,
    SampleCovariance,
    ZeroCorrelation,
)
from hdrbp.date import CalendarDate
from hdrbp.metric import (
    ExpectedShortfall,
    GeometricMeanReturn,
    MaxDrawdown,
    MeanCorrelation,
    MeanDiversificationRatio,
    MeanReturn,
    MeanRiskContributionGini,
    MeanVariance,
    MeanWeightGini,
    SharpeRatio,
    ValueAtRisk,
    Volatility,
)
from hdrbp.weight import (
    EqualRiskContribution,
    EqualWeight,
    MinimumCorrelation,
    MinimumVariance,
    MostDiversified,
    NaiveEqualRiskContribution,
)

# Data generation
SEED = 42
TIME_COUNT = 1_000
ASSET_COUNT = 10

# Contamination
CONTAMINATION_RATIO = 0.1
CONTAMINATION_SIZE = 100

# Backtest
ESTIMATION_SIZE = 12
HOLDING_SIZE = 2
REBALANCE_SCALE = "MS"
PORTFOLIO_SIZE = 5


def main():
    setup_logger()

    returns = generate_returns(TIME_COUNT, ASSET_COUNT, SEED)
    volumes = generate_volumes(TIME_COUNT, ASSET_COUNT, SEED)

    returns = contaminate(returns, CONTAMINATION_RATIO, CONTAMINATION_SIZE, SEED)
    volumes = contaminate(volumes, CONTAMINATION_RATIO, CONTAMINATION_SIZE, SEED)

    backtester = define_backtester(ESTIMATION_SIZE, HOLDING_SIZE, REBALANCE_SCALE, PORTFOLIO_SIZE)
    backtester.backtest(returns, covariates=volumes)

    print(backtester.metrics)


def setup_logger():
    logging.basicConfig(
        format="[{asctime}] [{levelname:<8}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.DEBUG,
    )

    logging.getLogger("numba").setLevel(logging.INFO)


def generate_returns(time_count, asset_count, seed=None):
    dates = generate_dates(time_count)
    tickers = generate_tickers(asset_count)
    return_values = generate_return_values(time_count, asset_count, seed)

    returns = pd.DataFrame(return_values, index=dates, columns=tickers)

    return returns


def generate_return_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)
    means = generator.uniform(0, 0.0001, size=asset_count)
    raw_covariances = generator.uniform(0, 0.01, size=(asset_count, asset_count))
    covariances = raw_covariances.T @ raw_covariances

    return_values = generator.multivariate_normal(means, covariances, size=time_count)

    return return_values


def generate_volumes(time_count, asset_count, seed=None):
    dates = generate_dates(time_count)
    tickers = generate_tickers(asset_count)
    volume_values = generate_volume_values(time_count, asset_count, seed)

    volumes = pd.DataFrame(volume_values, index=dates, columns=tickers)

    return volumes


def generate_volume_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)
    means = generator.uniform(1, 10, size=asset_count)
    covariances = np.diag(np.sqrt(means))

    volume_values = np.exp(generator.multivariate_normal(means, covariances, size=time_count))

    return volume_values


def generate_dates(time_count):
    return pd.date_range("01/01/2021", periods=time_count, freq="B")


def generate_tickers(asset_count):
    return [f"T{i}" for i in range(asset_count)]


def contaminate(data, ratio, size, seed=None):
    data = data.copy()
    values = data.values

    time_count, asset_count = values.shape
    period_count = time_count // size
    nan_time_count = size * period_count
    nan_asset_count = int(ratio * asset_count)

    times = np.arange(nan_time_count).reshape(-1, 1)
    times = np.repeat(times, repeats=nan_asset_count, axis=1)

    generator = np.random.default_rng(seed)
    assets = generator.choice(asset_count, size=(period_count, nan_asset_count))
    assets = np.repeat(assets, repeats=size, axis=0)

    values[times, assets] = np.nan

    return data


def define_backtester(estimation_size, holding_size, rebalance_scale, portfolio_size):
    rolling_window = RollingWindow(
        date_rule=CalendarDate(estimation_size, holding_size, rebalance_scale),
        asset_rule=LiquidAsset(portfolio_size),
    )

    strategies = Strategy.from_product(
        covariance_estimators=[
            EqualCorrelation(),
            EqualVariance(),
            LinearShrinkage(),
            RiskMetrics1994(),
            RiskMetrics2006(),
            SampleCovariance(),
            ZeroCorrelation(),
        ],
        weight_optimizers=[
            EqualRiskContribution(),
            EqualWeight(),
            MinimumCorrelation(),
            MinimumVariance(),
            MostDiversified(),
            NaiveEqualRiskContribution(),
        ],
    )

    metric_calculators = [
        ExpectedShortfall(),
        GeometricMeanReturn(),
        MaxDrawdown(),
        MeanCorrelation(),
        MeanDiversificationRatio(),
        MeanReturn(),
        MeanRiskContributionGini(),
        MeanVariance(),
        MeanWeightGini(),
        SharpeRatio(),
        ValueAtRisk(),
        Volatility(),
    ]

    backtester = Backtester(rolling_window, strategies, metric_calculators)

    return backtester


if __name__ == "__main__":
    main()
