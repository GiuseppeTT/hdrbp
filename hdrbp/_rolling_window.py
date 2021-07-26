import logging
from typing import Optional

import pandas as pd

from hdrbp._step import StepData, StepEstimationData, StepHoldingData
from hdrbp._util import basic_repr, basic_str
from hdrbp.asset import AssetRule
from hdrbp.date import DateRule

logger = logging.getLogger(__name__)


# TODO: unify extract, filter and select methods into, maybe, extract?
@basic_str
@basic_repr
class RollingWindow:
    def __init__(self, date_rule: DateRule, asset_rule: AssetRule) -> None:
        self._date_rule = date_rule
        self._asset_rule = asset_rule

    def count_possible_steps(self, returns: pd.DataFrame) -> int:
        dates = returns.index

        return self._date_rule.count_possible_steps(dates)

    def extract_data(
        self,
        index: int,
        returns: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
    ) -> StepData:
        logger.debug(f"{self}: Extracting data")

        estimation_data = self._extract_estimation_data(index, returns, covariates)
        holding_data = self._extract_holding_data(
            index, returns, covariates, estimation_data.assets
        )

        data = StepData(estimation_data, holding_data)

        return data

    def _extract_estimation_data(self, index, returns, covariates):
        dates, assets = self._extract_estimation_subset(index, returns, covariates)

        returns = returns.loc[dates, assets].values
        covariates = None if covariates is None else covariates.loc[dates, assets].values

        data = StepEstimationData(dates, assets, returns, covariates)

        return data

    def _extract_estimation_subset(self, index, returns, covariates):
        raw_dates = returns.index
        dates = self._date_rule.filter_estimation_dates(index, raw_dates)

        raw_returns = returns.loc[dates, :]
        raw_covariates = None if covariates is None else covariates.loc[dates, :]
        assets = self._asset_rule.select_assets(raw_returns, raw_covariates)

        return dates, assets

    def _extract_holding_data(self, index, returns, covariates, assets):
        dates = self._extract_holding_subset(index, returns)

        returns = returns.loc[dates, assets].values
        covariates = None if covariates is None else covariates.loc[dates, assets].values

        data = StepHoldingData(dates, assets, returns, covariates)

        return data

    def _extract_holding_subset(self, index, returns):
        raw_dates = returns.index
        dates = self._date_rule.filter_holding_dates(index, raw_dates)

        return dates
