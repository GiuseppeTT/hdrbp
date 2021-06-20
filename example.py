import logging

import pandas as pd
from numpy.random import default_rng

from hdrbp import Backtester, RollingWindow, Strategy
from hdrbp.asset import ValidAsset
from hdrbp.covariance import SampleCovariance
from hdrbp.date import TradingDate
from hdrbp.metric import MeanReturn
from hdrbp.weight import NaiveEqualRiskContribution

# Return generation
SEED = 42
TIME_COUNT = 1000
ASSET_COUNT = 10

# Backtest
ESTIMATION_SIZE = 100
HOLDING_SIZE = 10
PORTFOLIO_SIZE = 5


def main():
    setup_logger()
    returns = generate_returns(TIME_COUNT, ASSET_COUNT, SEED)
    backtester = define_backtester(ESTIMATION_SIZE, HOLDING_SIZE, PORTFOLIO_SIZE)

    backtester.backtest(returns)

    print(backtester.metrics)


def setup_logger():
    logging.basicConfig(
        format="[{asctime}] [{levelname:<8}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.DEBUG,
    )


def generate_returns(time_count, asset_count, seed=None):
    dates = pd.date_range("01/01/2021", periods=time_count, freq="B")
    tickers = [str(i) for i in range(asset_count)]
    return_values = generate_return_values(time_count, asset_count, seed)

    returns = pd.DataFrame(return_values, index=dates, columns=tickers)

    return returns


def generate_return_values(time_count, asset_count, seed):
    generator = default_rng(seed)
    means = generator.uniform(0, 0.0001, size=asset_count)
    raw_covariances = generator.uniform(0, 0.01, size=(asset_count, asset_count))
    covariances = raw_covariances.T @ raw_covariances
    return_values = generator.multivariate_normal(means, covariances, size=time_count)

    return return_values


def define_backtester(estimation_size, holding_size, portfolio_size):
    rolling_window = RollingWindow(
        date_rule=TradingDate(estimation_size, holding_size),
        asset_rule=ValidAsset(portfolio_size),
    )

    strategies = Strategy.from_product(
        covariance_estimators=[
            SampleCovariance(),
        ],
        weight_optimizers=[
            NaiveEqualRiskContribution(),
        ],
    )

    metric_calculators = [
        MeanReturn(),
    ]

    backtester = Backtester(rolling_window, strategies, metric_calculators)

    return backtester


if __name__ == "__main__":
    main()
