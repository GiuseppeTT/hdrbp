import logging
from abc import ABC, abstractmethod

import cvxopt
import numpy as np

from hdrbp._util import (
    CVXOPT_OPTIONS,
    basic_repr,
    basic_str,
    enforce_sum_one,
    extract_correlations,
    extract_volatilities,
    extract_weights,
)

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
        # Minimize variance = weights @ covariances @ weights
        # Subjected to weights >= 0 and sum(weights) = 1

        logger.debug(f"{self}: Optimizing weights")

        _, asset_count = covariances.shape

        P = cvxopt.matrix(covariances)
        q = cvxopt.matrix(0.0, (asset_count, 1))
        G = cvxopt.matrix(-np.eye(asset_count))
        h = cvxopt.matrix(0.0, (asset_count, 1))
        A = cvxopt.matrix(1.0, (1, asset_count))
        b = cvxopt.matrix(1.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b, options=CVXOPT_OPTIONS)
        weights = extract_weights(solution)

        return weights


class MinimumCorrelation(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        # Minimize "correlation" = weights @ correlations @ weights
        # Subjected to weights >= 0 and sum(weights) = 1

        logger.debug(f"{self}: Optimizing weights")

        correlations = extract_correlations(covariances)
        weights = MinimumVariance().optimize(correlations)

        return weights


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
