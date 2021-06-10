import logging
from abc import ABC, abstractmethod

import numpy as np

from hdrbp._util import basic_repr, basic_str, demean

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class CovarianceEstimator(ABC):
    def __hash__(self) -> int:
        return hash(repr(self))

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Estimating covariances")

        try:
            covariances = self._meaned_estimate(returns)
        except NotImplementedError:
            demeaned_returns = demean(returns, axis=0, keepdims=True)
            covariances = self._demeaned_estimate(demeaned_returns)

        return covariances

    @abstractmethod
    def _meaned_estimate(self, returns):
        pass

    @abstractmethod
    def _demeaned_estimate(self, returns):
        pass


class SampleCovariance(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        logger.debug(f"{self}: Demeaned estimating covariances")

        time_count, _ = returns.shape
        covariances = returns.T @ returns / time_count

        return covariances
