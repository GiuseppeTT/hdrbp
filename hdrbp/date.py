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
        # Basically an inversion of "start formula" in filter_holding_dates method
        break_dates = self._break_dates(dates)
        possible_count = (break_dates.size - self._estimation_size) // self._holding_size

        return possible_count

    def filter_estimation_dates(self, index: int, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        logger.debug(f"{self}: Filtering estimation dates")

        start = index * self._holding_size
        end = index * self._holding_size + self._estimation_size
        filtered_dates = self._filter_dates(dates, start, end)

        return filtered_dates

    def filter_holding_dates(self, index: int, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        logger.debug(f"{self}: Filtering holding dates")

        start = index * self._holding_size + self._estimation_size
        end = index * self._holding_size + self._estimation_size + self._holding_size
        filtered_dates = self._filter_dates(dates, start, end)

        return filtered_dates

    def _filter_dates(self, dates, start, end):
        break_dates = self._break_dates(dates)

        min_date = break_dates[start]

        # Last step may have incomplete holding period
        try:
            max_date = break_dates[end]
        except IndexError:
            max_date = pd.Timestamp.max

        is_filtered_dates = (min_date <= dates) & (dates < max_date)
        filtered_dates = dates[is_filtered_dates]

        return filtered_dates

    @abstractmethod
    def _break_dates(self, dates):
        pass


class TradingDate(DateRule):
    def _break_dates(self, dates):
        return dates


# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
class CalendarDate(DateRule):
    def __init__(self, estimation_size: int, holding_size: int, frequency: str = "D") -> None:
        super().__init__(estimation_size, holding_size)
        self._frequency = frequency

    def _break_dates(self, dates):
        return pd.date_range(dates.min(), dates.max(), freq=self._frequency)
