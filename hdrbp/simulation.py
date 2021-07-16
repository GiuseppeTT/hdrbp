from typing import Optional

import numpy as np
import pandas as pd

from hdrbp._util import count_digits


def generate_dates(
    time_count: int,
    start_date: str = "01/01/2000",
    frequency: str = "B",
) -> pd.DatetimeIndex:
    return pd.date_range(start_date, periods=time_count, freq=frequency)


def generate_tickers(asset_count: int) -> pd.Index:
    digit_count = count_digits(asset_count)

    return pd.Index(f"T{asset:0{digit_count}}" for asset in range(asset_count))


def generate_returns(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    time_count = dates.size
    asset_count = tickers.size
    return_values = _generate_return_values(time_count, asset_count, seed)

    returns = pd.DataFrame(return_values, index=dates, columns=tickers)

    return returns


# TODO: Make more convincing data.
# Generating return means is easy, just use a normal fit to sp500 data.
# The problem is generating return covariances.
# Maybe estimate volatilities from sp500 data using a gamma and estimate
# correlations eigenvalues and create a positive definite matrix from it
# (check scipy.stats.random_correlation). Later join them with
# hdrbp._util.build_covariances
def _generate_return_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)

    means = generator.uniform(0, 0.0001, size=asset_count)
    raw_covariances = generator.uniform(0, 0.01, size=(asset_count, asset_count))
    covariances = raw_covariances.T @ raw_covariances

    return_values = generator.multivariate_normal(means, covariances, size=time_count)

    return return_values


def generate_volumes(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    time_count = dates.size
    asset_count = tickers.size
    volume_values = _generate_volume_values(time_count, asset_count, seed)

    volumes = pd.DataFrame(volume_values, index=dates, columns=tickers)

    return volumes


# TODO: Make more convincing data.
def _generate_volume_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)

    means = generator.uniform(1, 10, size=asset_count)
    covariances = np.diag(np.sqrt(means))

    volume_values = np.exp(generator.multivariate_normal(means, covariances, size=time_count))

    return volume_values


def contaminate(
    data: pd.DataFrame,
    ratio: float,
    size: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
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
