import logging

import pandas as pd

from hdrbp._step import StepData, StepEstimationData, StepHoldingData
from hdrbp._util import basic_repr, basic_str
from hdrbp.asset import AssetRule
from hdrbp.date import DateRule

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class RollingWindow:
    def __init__(self, date_rule: DateRule, asset_rule: AssetRule) -> None:
        self._date_rule = date_rule
        self._asset_rule = asset_rule

    def count_possible_steps(self, returns: pd.DataFrame) -> int:
        dates = returns.index

        return self._date_rule.count_possible_steps(dates)

    def extract_data(self, index: int, returns: pd.DataFrame) -> StepData:
        logger.debug(f"{self}: Extracting data")

        estimation_data = self._extract_estimation_data(index, returns)
        holding_data = self._extract_holding_data(index, returns, estimation_data.assets)

        data = StepData(estimation_data, holding_data)

        return data

    def _extract_estimation_data(self, index, returns):
        raw_dates = returns.index
        dates = self._date_rule.filter_estimation_dates(index, raw_dates)
        raw_returns = returns.loc[dates, :]
        assets = self._asset_rule.select_assets(raw_returns)
        returns = returns.loc[dates, assets].values

        data = StepEstimationData(dates, assets, returns)

        return data

    def _extract_holding_data(self, index, returns, assets):
        raw_dates = returns.index
        dates = self._date_rule.filter_holding_dates(index, raw_dates)
        returns = returns.loc[dates, assets].values

        data = StepHoldingData(dates, assets, returns)

        return data
