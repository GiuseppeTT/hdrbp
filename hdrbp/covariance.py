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


class EqualVariance(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        logger.debug(f"{self}: Demeaned estimating covariances")

        time_count, asset_count = returns.shape
        global_variance = np.sum(returns**2) / (time_count * asset_count)
        covariances = global_variance * np.eye(asset_count)

        return covariances


class ZeroCorrelation(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        logger.debug(f"{self}: Demeaned estimating covariances")

        time_count, _ = returns.shape
        variances = np.sum(returns**2, axis=0) / time_count
        covariances = np.diag(variances)

        return covariances


class EqualCorrelation(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


class SampleCovariance(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        logger.debug(f"{self}: Demeaned estimating covariances")

        time_count, _ = returns.shape
        covariances = returns.T @ returns / time_count

        return covariances


class RiskMetrics1994(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


class RiskMetrics2006(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


class LinearShrinkage(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


class NonLinearShrinkage(CovarianceEstimator):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


# TODO: choose a better name?
# TODO: merge CCC and DCC into one class?
class GARCHCovariance(CovarianceEstimator, ABC):
    def _meaned_estimate(self, returns):
        raise NotImplementedError

    def _demeaned_estimate(self, returns):
        raise NotImplementedError


class ConstantConditionalCorrelation(GARCHCovariance):
    pass


class DynamicConditionalCorrelation(GARCHCovariance):
    pass
