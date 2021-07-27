import logging
from abc import ABC, abstractmethod

import cvxopt
import numpy as np
from numba import njit

from hdrbp._util import (
    CVXOPT_OPTIONS,
    ERROR_TOLERANCE,
    basic_repr,
    basic_str,
    enforce_unitary_sum,
    extract_correlations,
    extract_solution,
    extract_standard_deviations,
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
        weights = enforce_unitary_sum(weights)

        return weights


class EqualRiskContribution(WeightOptimizer):
    # https://dx.doi.org/10.2139/ssrn.2297383
    # https://dx.doi.org/10.2139/ssrn.2325255
    # https://dx.doi.org/10.2139/ssrn.1987770
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Optimizing weights")

        return _optimize_equal_risk_contribution(covariances)


@njit()
def _optimize_equal_risk_contribution(
    covariances: np.ndarray,
    max_iteration_count: int = 1_000,
    error_tolerance: float = ERROR_TOLERANCE,
) -> np.ndarray:
    _, asset_count = covariances.shape

    weights = np.ones(asset_count)
    weights = weights / np.sum(weights)
    mean_covariances = covariances @ weights
    portfolio_volatility = np.sqrt(weights @ covariances @ weights)
    for _ in range(max_iteration_count):
        for asset in range(asset_count):
            old_weight = weights[asset]
            new_weight = _update_weight(
                asset,
                old_weight,
                covariances,
                mean_covariances,
                portfolio_volatility,
                asset_count,
            )

            mean_covariances = _update_mean_covariances(
                asset,
                mean_covariances,
                covariances,
                old_weight,
                new_weight,
            )
            portfolio_volatility = _update_portfolio_volatility(
                asset,
                portfolio_volatility,
                covariances,
                weights,
                new_weight,
                old_weight,
            )

            weights[asset] = new_weight

        error = _compute_risk_contribution_error(weights, mean_covariances, asset_count)
        if error < error_tolerance:
            break

    weights = weights / np.sum(weights)

    return weights


@njit()
def _update_weight(
    asset,
    old_weight,
    covariances,
    mean_covariances,
    portfolio_volatility,
    asset_count,
):
    new_weight = (
        -(mean_covariances[asset] - old_weight * covariances[asset, asset])
        + np.sqrt(
            (mean_covariances[asset] - old_weight * covariances[asset, asset]) ** 2
            + 4 * covariances[asset, asset] * portfolio_volatility / asset_count
        )
    ) / (2 * covariances[asset, asset])

    return new_weight


@njit()
def _update_mean_covariances(asset, mean_covariances, covariances, old_weight, new_weight):
    return mean_covariances + covariances[:, asset] * (new_weight - old_weight)


@njit()
def _update_portfolio_volatility(
    asset, portfolio_volatility, covariances, weights, new_weight, old_weight
):
    portfolio_volatility = np.sqrt(
        portfolio_volatility ** 2
        + 2 * (new_weight - old_weight) * covariances[asset, :] @ weights
        + covariances[asset, asset] * (new_weight - old_weight) ** 2
    )

    return portfolio_volatility


@njit()
def _compute_risk_contribution_error(weights, mean_covariances, asset_count):
    risk_contributions = weights * mean_covariances
    risk_contributions = risk_contributions / np.sum(risk_contributions)
    risk_contribution_error = np.max(np.abs(risk_contributions - 1 / asset_count))

    return risk_contribution_error


class NaiveEqualRiskContribution(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Optimizing weights")

        volatilities = extract_standard_deviations(covariances)
        weights = enforce_unitary_sum(1 / volatilities)

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

        optimization_results = cvxopt.solvers.qp(P, q, G, h, A, b, options=CVXOPT_OPTIONS)
        weights = extract_solution(optimization_results)

        return weights


class MinimumCorrelation(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        # Minimize "correlation" = weights @ correlations @ weights
        # Subjected to weights >= 0 and sum(weights) = 1

        logger.debug(f"{self}: Optimizing weights")

        correlations = extract_correlations(covariances)
        weights = MinimumVariance().optimize(correlations)

        return weights


# TODO: find article that defines the QP formulation, or at least the lagrange of log
class MostDiversified(WeightOptimizer):
    # doi.org/10.3905/JPM.2008.35.1.40
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        # Minimize diversification_ratio = naive_volatility / volatility
        # Subjected to weights >= 0 and sum(weights) = 1
        # With naive_volatility = weights @ sqrt(diag(covariances))
        # and volatility = sqrt(weights @ covariances @ weights)

        logger.debug(f"{self}: Optimizing weights")

        _, asset_count = covariances.shape
        volatilities = extract_standard_deviations(covariances)

        P = cvxopt.matrix(covariances)
        q = cvxopt.matrix(0.0, (asset_count, 1))
        G = cvxopt.matrix(-np.eye(asset_count))
        h = cvxopt.matrix(0.0, (asset_count, 1))
        A = cvxopt.matrix(volatilities).T
        b = cvxopt.matrix(1.0)

        # By the optimization design, the solution is not guaranteed to sum to one
        optimization_results = cvxopt.solvers.qp(P, q, G, h, A, b, options=CVXOPT_OPTIONS)
        weights = extract_solution(optimization_results)
        weights = enforce_unitary_sum(weights)

        return weights


# TODO
class HierarchicalWeightOptimizer(WeightOptimizer, ABC):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HierarchicalEqualWeight(HierarchicalWeightOptimizer):
    pass


class HierarchicalEqualRiskContribution(HierarchicalWeightOptimizer):
    pass
