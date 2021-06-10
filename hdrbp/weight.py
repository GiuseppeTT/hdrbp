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


class InverseVolatility(WeightOptimizer):
    def optimize(self, covariances: np.ndarray) -> np.ndarray:
        logger.debug(f"{self}: Optimizing weights")

        volatilities = extract_volatilities(covariances)
        weights = enforce_sum_one(1 / volatilities)

        return weights
