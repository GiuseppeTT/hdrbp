import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd

from hdrbp._util import basic_repr, basic_str

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class AssetRule(ABC):
    def __init__(self, size: Optional[int] = None) -> None:
        self._size = size

    @abstractmethod
    def extract_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        pass


class ValidAsset(AssetRule):
    def extract_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        logger.debug(f"{self}: Extracting assets")

        valid_assets = _find_valid_assets(returns, covariates)
        size = _default_size(self._size, valid_assets)

        assets = valid_assets[:size]

        return assets


class RandomAsset(AssetRule):
    def __init__(self, size: Optional[int] = None, seed: Optional[int] = None) -> None:
        super().__init__(size)
        self._seed = seed

    def extract_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        logger.debug(f"{self}: Extracting assets")

        valid_assets = _find_valid_assets(returns, covariates)
        size = _default_size(self._size, valid_assets)

        generator = np.random.default_rng(self._seed)
        asset_ids = generator.choice(valid_assets.size, size=size, replace=False)
        asset_ids = np.sort(asset_ids)

        assets = valid_assets[asset_ids]

        return assets


class TopAsset(AssetRule):
    def __init__(self, size: Optional[int] = None, summarizer: Callable = np.mean) -> None:
        super().__init__(size)
        self._summarizer = summarizer

    def extract_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        logger.debug(f"{self}: Extracting assets")

        if covariates is None:
            raise ValueError("Covariates must be provided.")

        valid_assets = _find_valid_assets(returns, covariates)
        size = _default_size(self._size, valid_assets)

        valid_covariates = covariates.loc[:, valid_assets]
        summarized_covariates = valid_covariates.apply(self._summarizer, axis="rows")
        asset_ids = np.argsort(summarized_covariates.values)
        asset_ids = asset_ids[-size:]  # pylint: disable=invalid-unary-operand-type
        asset_ids = np.sort(asset_ids)

        assets = valid_assets[asset_ids]

        return assets


def _find_valid_assets(returns, covariates):
    if covariates is None:
        is_valid_assets = returns.notna().all()
    else:
        is_valid_assets = returns.notna().all() & covariates.notna().all()

    valid_assets = returns.columns[is_valid_assets]

    return valid_assets


def _default_size(size, valid_assets):
    if size is None:
        return valid_assets.size
    else:
        return size
