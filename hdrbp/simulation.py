import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import random_correlation

from hdrbp._util import build_covariances, count_digits, enforce_unitary_sum

logger = logging.getLogger(__name__)


def generate_dates(
    time_count: int,
    start_date: str = "01/01/2000",
    frequency: str = "B",
) -> pd.DatetimeIndex:
    logger.debug("Simulation: Generating dates")

    return pd.date_range(start_date, periods=time_count, freq=frequency)


def generate_assets(asset_count: int) -> pd.Index:
    logger.debug("Simulation: Generating assets")

    digit_count = count_digits(asset_count)

    return pd.Index(f"A{asset:0{digit_count}}" for asset in range(asset_count))


def generate_returns(
    dates: pd.DatetimeIndex,
    assets: pd.Index,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    logger.debug("Simulation: Generating returns")

    time_count = dates.size
    asset_count = assets.size
    return_values = _generate_return_values(time_count, asset_count, seed)

    returns = pd.DataFrame(return_values, index=dates, columns=assets)

    return returns


def _generate_return_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)

    means = _generate_means(generator, asset_count, location=0.0005, scale=0.0005)
    volatilities = _generate_standard_deviations(generator, asset_count, shape=16, scale=1 / 800)
    correlations = _generate_correlations(generator, asset_count, location=-5, scale=1)
    covariances = build_covariances(volatilities, correlations)

    return_values = generator.multivariate_normal(means, covariances, size=time_count)
    return_values = np.expm1(return_values)

    return return_values


def generate_volumes(
    dates: pd.DatetimeIndex,
    assets: pd.Index,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    logger.debug("Simulation: Generating volumes")

    time_count = dates.size
    asset_count = assets.size
    volume_values = _generate_volume_values(time_count, asset_count, seed)

    volumes = pd.DataFrame(volume_values, index=dates, columns=assets)

    return volumes


def _generate_volume_values(time_count, asset_count, seed=None):
    generator = np.random.default_rng(seed)

    means = _generate_means(generator, asset_count, location=15, scale=2)
    standard_deviations = _generate_standard_deviations(
        generator, asset_count, shape=25, scale=1 / 60
    )
    correlations = _generate_correlations(generator, asset_count, location=-7.5, scale=1.5)
    covariances = build_covariances(standard_deviations, correlations)

    volume_values = generator.multivariate_normal(means, covariances, size=time_count)
    volume_values = np.exp(volume_values)

    return volume_values


def _generate_means(generator, asset_count, location, scale):
    return generator.normal(location, scale, size=asset_count)


def _generate_standard_deviations(generator, asset_count, shape, scale):
    return generator.gamma(shape, scale, size=asset_count)


def _generate_correlations(generator, asset_count, location, scale):
    eigen_values = generator.normal(location, scale, size=asset_count)
    eigen_values = np.exp(eigen_values)
    eigen_values = asset_count * enforce_unitary_sum(eigen_values)

    return random_correlation.rvs(eigen_values)


def contaminate(
    data: pd.DataFrame,
    ratio: float,
    size: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    logger.debug("Simulation: Contaminating data")

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
