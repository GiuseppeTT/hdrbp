import logging
from abc import ABC, abstractmethod

import pandas as pd

from hdrbp._util import basic_repr, basic_str

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class DateRule(ABC):
    def __init__(self, estimation_size: int, holding_size: int) -> None:
        self._estimation_size = estimation_size
        self._holding_size = holding_size

    def count_possible_steps(self, dates: pd.DatetimeIndex) -> int:
        # Basically an inversion of "start formula" in extract_holding_dates method
        break_dates = self._break_dates(dates)
        possible_count = (break_dates.size - self._estimation_size) // self._holding_size

        return possible_count

    def extract_estimation_dates(self, index: int, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        logger.debug(f"{self}: Extracting estimation dates")

        start = index * self._holding_size
        end = index * self._holding_size + self._estimation_size
        extracted_dates = self._extract_dates(dates, start, end)

        return extracted_dates

    def extract_holding_dates(self, index: int, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        logger.debug(f"{self}: Extracting holding dates")

        start = index * self._holding_size + self._estimation_size
        end = index * self._holding_size + self._estimation_size + self._holding_size
        extracted_dates = self._extract_dates(dates, start, end)

        return extracted_dates

    def _extract_dates(self, dates, start, end):
        break_dates = self._break_dates(dates)

        min_date = break_dates[start]

        # Last step may have incomplete holding period
        try:
            max_date = break_dates[end]
        except IndexError:
            max_date = pd.Timestamp.max

        is_extracted_dates = (min_date <= dates) & (dates < max_date)
        extracted_dates = dates[is_extracted_dates]

        return extracted_dates

    @abstractmethod
    def _break_dates(self, dates):
        pass


class TradingDate(DateRule):
    def _break_dates(self, dates):
        return dates


# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
class CalendarDate(DateRule):
    def __init__(self, estimation_size: int, holding_size: int, rebalance_scale: str = "D") -> None:
        super().__init__(estimation_size, holding_size)
        self._rebalance_scale = rebalance_scale

    def _break_dates(self, dates):
        return pd.date_range(dates.min(), dates.max(), freq=self._rebalance_scale)
