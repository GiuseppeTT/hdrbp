import logging

import pandas as pd
from numpy.random import default_rng

from hdrbp import Backtester, RollingWindow, Strategy
from hdrbp.asset import ValidAsset
from hdrbp.covariance import SampleCovariance
from hdrbp.date import TradingDate
from hdrbp.metric import MeanReturn
from hdrbp.weight import InverseVolatility

# Return generation
SEED = 42
TIME_COUNT = 1000
ASSET_COUNT = 10

# Backtest
ESTIMATION_SIZE = 100
HOLDING_SIZE = 10
PORTFOLIO_SIZE = 5

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.DEBUG,
)


def main():
    returns = generate_returns()
    backtester = define_backtester()

    backtester.backtest(returns)

    print(backtester.metrics)


def generate_returns():
    dates = pd.date_range("01/01/2021", periods=TIME_COUNT, freq="B")
    tickers = [str(i) for i in range(ASSET_COUNT)]
    return_values = generate_return_values()

    returns = pd.DataFrame(return_values, index=dates, columns=tickers)

    return returns


def generate_return_values():
    generator = default_rng(seed=SEED)
    means = generator.uniform(0, 0.0001, size=ASSET_COUNT)
    raw_covariances = generator.uniform(0, 0.01, size=(ASSET_COUNT, ASSET_COUNT))
    covariances = raw_covariances.T @ raw_covariances
    return_values = generator.multivariate_normal(means, covariances, size=TIME_COUNT)

    return return_values


def define_backtester():
    rolling_window = RollingWindow(
        date_rule=TradingDate(estimation_size=ESTIMATION_SIZE, holding_size=HOLDING_SIZE),
        asset_rule=ValidAsset(size=PORTFOLIO_SIZE),
    )

    strategies = Strategy.from_product(
        covariance_estimators=[
            SampleCovariance(),
        ],
        weight_optimizers=[
            InverseVolatility(),
        ],
    )

    metric_calculators = [
        MeanReturn(),
    ]

    backtester = Backtester(rolling_window, strategies, metric_calculators)

    return backtester


if __name__ == "__main__":
    main()
