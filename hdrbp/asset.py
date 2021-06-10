import logging
from abc import ABC, abstractmethod

import pandas as pd

from hdrbp._util import basic_repr, basic_str

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class AssetRule(ABC):
    def __init__(self, size: int) -> None:
        self._size = size

    @abstractmethod
    def select_assets(self, returns: pd.DataFrame) -> pd.Index:
        pass

    def _find_valid_assets(self, returns):
        is_valid_assets = returns.notna().all()
        valid_assets = returns.columns[is_valid_assets]

        return valid_assets


class ValidAsset(AssetRule):
    def select_assets(self, returns: pd.DataFrame) -> pd.Index:
        logger.debug(f"{self}: Selecting assets")

        valid_assets = self._find_valid_assets(returns)
        assets = valid_assets[: self._size]

        return assets
