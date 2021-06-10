import logging
from abc import ABC, abstractmethod

import numpy as np

from hdrbp._util import basic_repr, basic_str, enforce_sum_one, extract_volatilities

logger = logging.getLogger(__name__)


@basic_str
@basic_repr
class WeightOptimizer(ABC):
    @abstractmethod
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        pass


class EqualWeight(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Optimizing weights")

        _, asset_count = covariances.shape
        weights = np.ones(asset_count)
        weights = enforce_sum_one(weights)

        return weights


class EqualRiskContribution(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NaiveEqualRiskContribution(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Optimizing weights")

        volatilities = extract_volatilities(covariances)
        weights = enforce_sum_one(1 / volatilities)

        return weights


class MinimumVariance(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MinimumCorrelation(MinimumVariance):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MostDiversified(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HierarchicalWeightOptimizer(WeightOptimizer, ABC):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HierarchicalEqualWeight(HierarchicalWeightOptimizer):
    pass


class HierarchicalEqualRiskContribution(HierarchicalWeightOptimizer):
    pass
