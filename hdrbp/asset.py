import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from hdrbp._util import basic_repr, basic_str

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class AssetRule(ABC):
    def __init__(self, size: Optional[int] = None) -> None:
        self._size = size

    @abstractmethod
    def select_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        pass


class ValidAsset(AssetRule):
    def select_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        logger.debug(f"{self}: Selecting assets")

        valid_assets = self._find_valid_assets(returns)
        assets = valid_assets[: self._size]

        return assets

    @staticmethod
    def _find_valid_assets(returns):
        is_valid_assets = returns.notna().all()
        valid_assets = returns.columns[is_valid_assets]

        return valid_assets


class LiquidAsset(AssetRule):
    def select_assets(
        self,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        logger.debug(f"{self}: Selecting assets")

        if covariates is None:
            raise ValueError("Asset volumes (covariates argument) must be provided.")

        volumes = covariates

        valid_assets = self._find_valid_assets(returns, volumes)
        mean_volumes = volumes.loc[:, valid_assets].mean(axis="rows")

        size = valid_assets.size if self._size is None else self._size
        top_mean_volumes = mean_volumes.nlargest(size)

        assets = top_mean_volumes.index.sort_values()

        return assets

    @staticmethod
    def _find_valid_assets(returns, volumes):
        is_valid_assets = returns.notna().all() & volumes.notna().all()
        valid_assets = returns.columns[is_valid_assets]

        return valid_assets
