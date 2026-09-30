import numpy as np


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""

    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error."""

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return 0.0

    return float(
        np.sum(np.abs(actual - predicted)) / denominator
    )