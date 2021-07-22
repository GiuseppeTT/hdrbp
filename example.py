import logging

from hdrbp import Backtester, RollingWindow, Strategy
from hdrbp.asset import LiquidAsset
from hdrbp.covariance import (
    EqualCorrelation,
    EqualVariance,
    LinearShrinkage,
    ExponentialWeighted,
    ExponentialWeightedMixture,
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
from hdrbp.simulation import (
    contaminate,
    generate_dates,
    generate_returns,
    generate_tickers,
    generate_volumes,
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

    returns, volumes = generate_data(
        TIME_COUNT, ASSET_COUNT, CONTAMINATION_RATIO, CONTAMINATION_SIZE, SEED
    )
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


def generate_data(time_count, asset_count, contamination_ratio, contamination_size, seed=None):
    dates = generate_dates(time_count)
    tickers = generate_tickers(asset_count)

    returns = generate_returns(dates, tickers, seed)
    volumes = generate_volumes(dates, tickers, seed)

    returns = contaminate(returns, contamination_ratio, contamination_size, seed)
    volumes = contaminate(volumes, contamination_ratio, contamination_size, seed)

    return returns, volumes


def define_backtester(estimation_size, holding_size, rebalance_scale, portfolio_size):
    rolling_window = RollingWindow(
        date_rule=CalendarDate(estimation_size, holding_size, rebalance_scale),
        asset_rule=LiquidAsset(portfolio_size),
    )

    strategies = Strategy.from_product(
        covariance_estimators=[
            EqualCorrelation(),
            EqualVariance(),
            ExponentialWeighted(),
            ExponentialWeightedMixture(),
            LinearShrinkage(),
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
